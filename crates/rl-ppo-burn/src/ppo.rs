use std::time::Instant;

use abm_gym::{EpisodeSummary, ExecutionConfig, OBS_DIM, Observation, SliceActions, VecEnv};
use burn::module::Module;
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::{Device, Int, Tensor, TensorData};
use market_agents::{Rng, derive_seed};
use serde::{Deserialize, Serialize};

use crate::model::{ActorCritic, ModelConfig};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PpoConfig {
    pub n_envs: usize,
    pub iterations: usize,
    pub epochs: usize,
    pub minibatch_envs: usize,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub clip: f64,
    pub value_coef: f64,
    pub entropy_coef: f64,
    pub learning_rate: f64,
    pub max_grad_norm: f32,
    pub reward_scale: f64,
    pub target_kl: f64,
    pub hidden: usize,
    pub seed: u64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            n_envs: 128,
            iterations: 200,
            epochs: 4,
            minibatch_envs: 32,
            gamma: 1.0,
            gae_lambda: 0.95,
            clip: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            learning_rate: 3e-4,
            max_grad_norm: 1.0,
            reward_scale: 0.1,
            target_kl: 0.03,
            hidden: 64,
            seed: 7,
        }
    }
}

/// Fixed-horizon rollout of `n` environments for at most `len` steps, stored env-major
/// (`index = env * len + t`). Steps after termination carry `mask = 0`.
#[derive(Clone, Debug)]
pub struct Rollout {
    pub n: usize,
    pub len: usize,
    pub obs: Vec<f32>,
    pub actions: Vec<i64>,
    pub logp: Vec<f32>,
    pub values: Vec<f32>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub mask: Vec<f32>,
    pub summaries: Vec<EpisodeSummary>,
}

pub(crate) fn obs_tensor(obs: &[Observation], device: &Device) -> Tensor<2> {
    let flat: Vec<f32> = obs.iter().flat_map(|o| o.iter().copied()).collect();
    Tensor::from_data(TensorData::new(flat, [obs.len(), OBS_DIM]), device)
}

pub(crate) fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_into_vec_as::<f32>().expect("float tensor")
}

/// Draws from `softmax(logits)` (or takes the mode when `greedy`) and returns the action with
/// its log-probability.
pub fn sample_row(logits: &[f32], rng: &mut Rng, greedy: bool) -> (usize, f32) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f64> = logits.iter().map(|l| ((l - max) as f64).exp()).collect();
    let z: f64 = w.iter().sum();
    let a = if greedy {
        logits
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |acc, (i, &l)| if l > acc.1 { (i, l) } else { acc })
            .0
    } else {
        let mut u = rng.uniform() * z;
        let mut pick = w.len() - 1;
        for (i, wi) in w.iter().enumerate() {
            if u < *wi {
                pick = i;
                break;
            }
            u -= wi;
        }
        pick
    };
    (a, (w[a] / z).ln() as f32)
}

pub fn collect(
    model: &ActorCritic,
    device: &Device,
    env: &mut VecEnv,
    actions: &SliceActions,
    seeds: &[u64],
    rng: &mut Rng,
    reward_scale: f64,
) -> Rollout {
    let n = env.len();
    let len = env.config().n_steps;
    let na = actions.len();
    let mut r = Rollout {
        n,
        len,
        obs: vec![0.0; n * len * OBS_DIM],
        actions: vec![0; n * len],
        logp: vec![0.0; n * len],
        values: vec![0.0; n * len],
        rewards: vec![0.0; n * len],
        dones: vec![false; n * len],
        mask: vec![0.0; n * len],
        summaries: Vec::new(),
    };
    let mut obs = env.reset(seeds);
    let mut h = Tensor::<2>::zeros([n, model.hidden()], device);
    for t in 0..len {
        let active: Vec<bool> = env.dones().iter().map(|d| !d).collect();
        let ctx = env.contexts();
        let (logits, value, next) = model.step(obs_tensor(&obs, device), h);
        h = next;
        let logits = to_vec(logits);
        let value = to_vec(value);
        let mut qty = vec![0u64; n];
        for i in (0..n).filter(|&i| active[i]) {
            let (a, lp) = sample_row(&logits[i * na..(i + 1) * na], rng, false);
            let k = i * len + t;
            r.obs[k * OBS_DIM..(k + 1) * OBS_DIM].copy_from_slice(&obs[i]);
            r.actions[k] = a as i64;
            r.logp[k] = lp;
            r.values[k] = value[i];
            r.mask[k] = 1.0;
            qty[i] = actions.quantity(a, &ctx[i]);
        }
        let out = env.step(&qty);
        for i in (0..n).filter(|&i| active[i]) {
            let k = i * len + t;
            r.rewards[k] = (out[i].reward * reward_scale) as f32;
            r.dones[k] = out[i].done;
        }
        if env.all_done() {
            break;
        }
        obs = env.observations();
    }
    r.summaries = env.summaries();
    r
}

/// Generalised advantage estimation `A_t = sum_l (gamma lambda)^l delta_{t+l}` with
/// `delta_t = r_t + gamma V_{t+1} (1 - done_t) - V_t`; returns `(A, A + V)`.
pub fn gae(r: &Rollout, gamma: f64, lambda: f64) -> (Vec<f32>, Vec<f32>) {
    let mut adv = vec![0.0f32; r.n * r.len];
    let mut ret = vec![0.0f32; r.n * r.len];
    for i in 0..r.n {
        let (mut next_adv, mut next_v) = (0.0f64, 0.0f64);
        for t in (0..r.len).rev() {
            let k = i * r.len + t;
            if r.mask[k] == 0.0 {
                next_adv = 0.0;
                next_v = 0.0;
                continue;
            }
            let live = if r.dones[k] { 0.0 } else { 1.0 };
            let v = r.values[k] as f64;
            let delta = r.rewards[k] as f64 + gamma * next_v * live - v;
            next_adv = delta + gamma * lambda * live * next_adv;
            adv[k] = next_adv as f32;
            ret[k] = (next_adv + v) as f32;
            next_v = v;
        }
    }
    (adv, ret)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UpdateStats {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub approx_kl: f64,
    pub clip_fraction: f64,
    pub epochs: usize,
}

fn shuffle(v: &mut [usize], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        v.swap(i, j);
    }
}

/// Clipped-surrogate PPO update over whole sequences (recurrent state recomputed from zero):
/// `L = -E[min(rho A, clip(rho, 1 - eps, 1 + eps) A)] + c_v E[(V - R)^2] - c_H E[H(pi)]`,
/// expectations taken over unmasked steps. Epochs stop early once the mean of the
/// `(rho - 1) - ln rho` KL estimator exceeds `1.5 * target_kl`.
#[allow(clippy::too_many_arguments)]
pub fn update(
    mut model: ActorCritic,
    optim: &mut ModuleOptimizer,
    device: &Device,
    r: &Rollout,
    adv: &[f32],
    ret: &[f32],
    cfg: &PpoConfig,
    lr: f64,
    rng: &mut Rng,
) -> (ActorCritic, UpdateStats) {
    let valid: Vec<f64> = adv
        .iter()
        .zip(&r.mask)
        .filter(|(_, m)| **m > 0.0)
        .map(|(a, _)| *a as f64)
        .collect();
    let mean = valid.iter().sum::<f64>() / valid.len() as f64;
    let std = (valid.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / valid.len() as f64).sqrt();
    let adv_n: Vec<f32> = adv
        .iter()
        .zip(&r.mask)
        .map(|(a, m)| if *m > 0.0 { ((*a as f64 - mean) / (std + 1e-8)) as f32 } else { 0.0 })
        .collect();

    let l = r.len;
    let mut order: Vec<usize> = (0..r.n).collect();
    let mut stats = UpdateStats::default();
    let mut batches = 0usize;
    for epoch in 0..cfg.epochs {
        shuffle(&mut order, rng);
        let mut epoch_kl = 0.0;
        let mut epoch_batches = 0usize;
        for mb in order.chunks(cfg.minibatch_envs.max(1)) {
            let b = mb.len();
            let mut obs = Vec::with_capacity(b * l * OBS_DIM);
            let mut act = Vec::with_capacity(b * l);
            let mut old = Vec::with_capacity(b * l);
            let mut a_mb = Vec::with_capacity(b * l);
            let mut ret_mb = Vec::with_capacity(b * l);
            let mut mask = Vec::with_capacity(b * l);
            for &i in mb {
                let s = i * l;
                obs.extend_from_slice(&r.obs[s * OBS_DIM..(s + l) * OBS_DIM]);
                act.extend_from_slice(&r.actions[s..s + l]);
                old.extend_from_slice(&r.logp[s..s + l]);
                a_mb.extend_from_slice(&adv_n[s..s + l]);
                ret_mb.extend_from_slice(&ret[s..s + l]);
                mask.extend_from_slice(&r.mask[s..s + l]);
            }
            let count = mask.iter().sum::<f32>().max(1.0);
            let obs_t = Tensor::<3>::from_data(TensorData::new(obs, [b, l, OBS_DIM]), device);
            let act_t = Tensor::<3, Int>::from_data(TensorData::new(act, [b, l, 1]), device);
            let old_t = Tensor::<2>::from_data(TensorData::new(old.clone(), [b, l]), device);
            let adv_t = Tensor::<2>::from_data(TensorData::new(a_mb, [b, l]), device);
            let ret_t = Tensor::<2>::from_data(TensorData::new(ret_mb, [b, l]), device);
            let mask_t = Tensor::<2>::from_data(TensorData::new(mask.clone(), [b, l]), device);

            let (logits, values) = model.forward_sequence(obs_t);
            let logp_all = log_softmax(logits, 2);
            let new_logp = logp_all.clone().gather(2, act_t).reshape([b, l]);
            let entropy = (logp_all.clone().exp() * logp_all).sum_dim(2).reshape([b, l]).neg();
            let ratio = (new_logp.clone() - old_t).exp();
            let s1 = ratio.clone() * adv_t.clone();
            let s2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv_t;
            let policy_loss = (s1.min_pair(s2) * mask_t.clone()).sum().div_scalar(count).neg();
            let value_loss = ((values - ret_t).powi_scalar(2) * mask_t.clone())
                .sum()
                .div_scalar(count);
            let ent = (entropy * mask_t).sum().div_scalar(count);
            let loss = policy_loss.clone() + value_loss.clone().mul_scalar(cfg.value_coef)
                - ent.clone().mul_scalar(cfg.entropy_coef);

            let new_lp = to_vec(new_logp.detach());
            let (mut kl, mut clipped) = (0.0f64, 0.0f64);
            for k in 0..new_lp.len() {
                if mask[k] > 0.0 {
                    let lr_k = (new_lp[k] - old[k]) as f64;
                    let rho = lr_k.exp();
                    kl += (rho - 1.0) - lr_k;
                    if (rho - 1.0).abs() > cfg.clip {
                        clipped += 1.0;
                    }
                }
            }
            kl /= count as f64;
            stats.policy_loss += to_vec(policy_loss.detach())[0] as f64;
            stats.value_loss += to_vec(value_loss.detach())[0] as f64;
            stats.entropy += to_vec(ent.detach())[0] as f64;
            stats.approx_kl += kl;
            stats.clip_fraction += clipped / count as f64;
            batches += 1;
            epoch_kl += kl;
            epoch_batches += 1;

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
        }
        stats.epochs = epoch + 1;
        if epoch_kl / epoch_batches as f64 > 1.5 * cfg.target_kl {
            break;
        }
    }
    let nb = batches.max(1) as f64;
    stats.policy_loss /= nb;
    stats.value_loss /= nb;
    stats.entropy /= nb;
    stats.approx_kl /= nb;
    stats.clip_fraction /= nb;
    (model, stats)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IterationLog {
    pub iteration: usize,
    pub episodes: usize,
    pub mean_shortfall_bps: f64,
    pub std_shortfall_bps: f64,
    pub mean_objective_bps: f64,
    pub mean_multiplier: f64,
    pub learning_rate: f64,
    pub update: UpdateStats,
    pub rollout_secs: f64,
    pub update_secs: f64,
}

pub struct Trainer {
    pub ppo: PpoConfig,
    pub exec: ExecutionConfig,
    pub actions: SliceActions,
    pub model_config: ModelConfig,
    device: Device,
    infer_device: Device,
    model: Option<ActorCritic>,
    optim: ModuleOptimizer,
    rng: Rng,
    env: VecEnv,
    iteration: usize,
}

impl Trainer {
    pub fn new(ppo: PpoConfig, exec: ExecutionConfig, actions: SliceActions, base: Device) -> Self {
        base.seed(ppo.seed);
        let device = base.clone().autodiff();
        let model_config = ModelConfig {
            obs_dim: OBS_DIM,
            hidden: ppo.hidden,
            n_actions: actions.len(),
        };
        let model = model_config.init(&device);
        let optim = AdamConfig::new()
            .with_epsilon(1e-5)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(ppo.max_grad_norm)))
            .init();
        Self {
            rng: Rng::new(derive_seed(ppo.seed, 0x5a)),
            env: VecEnv::new(exec.clone(), ppo.n_envs),
            ppo,
            exec,
            actions,
            model_config,
            infer_device: base,
            device,
            model: Some(model),
            optim,
            iteration: 0,
        }
    }

    pub fn inference_model(&self) -> ActorCritic {
        self.model.as_ref().expect("model present").valid()
    }

    pub fn inference_device(&self) -> &Device {
        &self.infer_device
    }

    pub fn train_seeds(&self, iteration: usize) -> Vec<u64> {
        (0..self.ppo.n_envs)
            .map(|i| derive_seed(self.ppo.seed ^ 0x7121_9000, (iteration * self.ppo.n_envs + i) as u64))
            .collect()
    }

    pub fn iteration(&mut self) -> IterationLog {
        let t0 = Instant::now();
        let seeds = self.train_seeds(self.iteration);
        let infer = self.inference_model();
        let rollout = collect(
            &infer,
            &self.infer_device,
            &mut self.env,
            &self.actions,
            &seeds,
            &mut self.rng,
            self.ppo.reward_scale,
        );
        let rollout_secs = t0.elapsed().as_secs_f64();
        let (adv, ret) = gae(&rollout, self.ppo.gamma, self.ppo.gae_lambda);
        let frac = self.iteration as f64 / self.ppo.iterations.max(1) as f64;
        let lr = self.ppo.learning_rate * (1.0 - frac).max(0.05);
        let t1 = Instant::now();
        let model = self.model.take().expect("model present");
        let (model, update) = update(
            model,
            &mut self.optim,
            &self.device,
            &rollout,
            &adv,
            &ret,
            &self.ppo,
            lr,
            &mut self.rng,
        );
        self.model = Some(model);
        let update_secs = t1.elapsed().as_secs_f64();

        let n = rollout.summaries.len() as f64;
        let is: Vec<f64> = rollout.summaries.iter().map(|s| s.shortfall_bps).collect();
        let mean = is.iter().sum::<f64>() / n;
        let std = (is.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let (mut msum, mut mcount) = (0.0, 0.0);
        for (a, m) in rollout.actions.iter().zip(&rollout.mask) {
            if *m > 0.0 {
                msum += self.actions.multipliers[*a as usize];
                mcount += 1.0;
            }
        }
        let log = IterationLog {
            iteration: self.iteration,
            episodes: rollout.summaries.len(),
            mean_shortfall_bps: mean,
            std_shortfall_bps: std,
            mean_objective_bps: rollout.summaries.iter().map(|s| s.objective_bps()).sum::<f64>() / n,
            mean_multiplier: msum / mcount,
            learning_rate: lr,
            update,
            rollout_secs,
            update_secs,
        };
        self.iteration += 1;
        log
    }
}
