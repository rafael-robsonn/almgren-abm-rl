use burn::module::Module;
use burn::nn::{Gru, GruConfig, Initializer, Linear, LinearConfig};
use burn::tensor::{Device, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub obs_dim: usize,
    pub hidden: usize,
    pub n_actions: usize,
}

/// Recurrent actor-critic: `e_t = tanh(W_e o_t + b_e)`, `h_t = GRU(e_t, h_{t-1})`,
/// policy logits `W_pi h_t + b_pi`, value `w_v . h_t + b_v`. Actor and critic share the
/// encoder and the recurrent state.
#[derive(Module, Debug)]
pub struct ActorCritic {
    encoder: Linear,
    gru: Gru,
    actor: Linear,
    critic: Linear,
    hidden: usize,
}

impl ModelConfig {
    pub fn init(&self, device: &Device) -> ActorCritic {
        ActorCritic {
            encoder: LinearConfig::new(self.obs_dim, self.hidden).init(device),
            gru: GruConfig::new(self.hidden, self.hidden, true).init(device),
            actor: LinearConfig::new(self.hidden, self.n_actions)
                .with_initializer(Initializer::Uniform {
                    min: -1e-2,
                    max: 1e-2,
                })
                .init(device),
            critic: LinearConfig::new(self.hidden, 1).init(device),
            hidden: self.hidden,
        }
    }
}

impl ActorCritic {
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Full-sequence pass from a zero state: `obs [B, L, D] -> (logits [B, L, A], values [B, L])`.
    pub fn forward_sequence(&self, obs: Tensor<3>) -> (Tensor<3>, Tensor<2>) {
        let [b, l, _] = obs.dims();
        let x = self.encoder.forward(obs).tanh();
        let h = self.gru.forward(x, None);
        let logits = self.actor.forward(h.clone());
        let values = self.critic.forward(h).reshape([b, l]);
        (logits, values)
    }

    /// One recurrent step: `(obs [B, D], h [B, H]) -> (logits [B, A], value [B], h' [B, H])`.
    pub fn step(&self, obs: Tensor<2>, state: Tensor<2>) -> (Tensor<2>, Tensor<1>, Tensor<2>) {
        let [b, d] = obs.dims();
        let x = self.encoder.forward(obs.reshape([b, 1, d])).tanh();
        let h = self.gru.forward(x, Some(state)).reshape([b, self.hidden]);
        let logits = self.actor.forward(h.clone());
        let value = self.critic.forward(h.clone()).reshape([b]);
        (logits, value, h)
    }
}
