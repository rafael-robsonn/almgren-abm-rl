use abm_gym::{Observation, Policy, SliceActions, StepContext};
use burn::tensor::{Device, Tensor};
use market_agents::Rng;
use matching_engine::Qty;

use crate::model::ActorCritic;
use crate::ppo::{obs_tensor, sample_row, to_vec};

/// Trained recurrent policy behind the batched [`Policy`] interface; the GRU state is carried
/// across steps and reset per batch of episodes.
pub struct PpoPolicy {
    model: ActorCritic,
    device: Device,
    actions: SliceActions,
    greedy: bool,
    rng: Rng,
    state: Option<Tensor<2>>,
    label: String,
}

impl PpoPolicy {
    pub fn new(model: ActorCritic, device: Device, actions: SliceActions, greedy: bool, seed: u64) -> Self {
        Self {
            model,
            device,
            actions,
            greedy,
            rng: Rng::new(seed),
            state: None,
            label: if greedy { "PPO-GRU (mode)".into() } else { "PPO-GRU (sampled)".into() },
        }
    }
}

impl Policy for PpoPolicy {
    fn name(&self) -> String {
        self.label.clone()
    }

    fn reset(&mut self, n_envs: usize) {
        self.state = Some(Tensor::zeros([n_envs, self.model.hidden()], &self.device));
    }

    fn act(&mut self, obs: &[Observation], ctx: &[StepContext], active: &[bool]) -> Vec<Qty> {
        let h = self.state.take().expect("reset before acting");
        let (logits, _, next) = self.model.step(obs_tensor(obs, &self.device), h);
        self.state = Some(next);
        let logits = to_vec(logits);
        let na = self.actions.len();
        (0..obs.len())
            .map(|i| {
                if !active[i] {
                    return 0;
                }
                let (a, _) = sample_row(&logits[i * na..(i + 1) * na], &mut self.rng, self.greedy);
                self.actions.quantity(a, &ctx[i])
            })
            .collect()
    }
}
