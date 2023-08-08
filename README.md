# Icy Q Learners

## Creating Lilith

* Lilith represents the simplest baseline one can conceive
* It is used to bootrstrap the training of other agent and as an "ELO anchor" to judge the relative improvements of various architectural changes
* It will also be used as an stronger and training-independent opponent during self-play of other agents

### Architecture
* Basic MLP with no layer-norm:
    * 2x Hidden-size 256, ReLU
    * Dueling: Two heads with one 256 ReLU hidden layer and one final linear layer

* No double-q-learning
* No PER
* No cosine annealing
* No nstep loss

The version created by the procedure below will be called *lilith-weak*.

*lilith-strong* is produced using the "basic-schedule" to train a new lilith-model, i.e. bootstrapped replay buffer and longer self-play.

### Training
* Eps-Decay: 1 million
* 3 million frames of training:
    - First 500k against weak, then strong is added
    - After 1.5M we start self-play
    - Each 100k steps (roughly 500-700 games) the current version is added to the pool (p=5)
    - We keep 5 versions around 


# Basic Training Schedule (Ablation)

Train for 5M (~4h)
* Eps-Decay: 2M
* Cosine Annealing
1. Bootstrap Replay Buffer with Lilith (150k)
2. Train for 500k against weak, then add strong
3. Add lilith-weak at 1M (p=5)
4. Start self-training add 2M
    - Add one copy every 200k frames (p=5)
    - 8 versions total