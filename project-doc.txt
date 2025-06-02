**MageZero: A Deck-Local AI Framework for Magic: The Gathering**

---

### 1. High-Level Philosophy

MageZero is not a reinforcement learning (RL) agent in itself. It is a framework for training and managing deck-specific RL agents for Magic: The Gathering (MTG). Rather than attempting to generalize across the entire game with a monolithic model, MageZero decomposes MTG into smaller, more tractable subgames. Each 60–120 card deck is treated as a self-contained "bubble" that can be mastered independently using focused, lightweight RL techniques.

This approach reframes the challenge of MTG AI from universal mastery to local optimization. By training agents within constrained, well-defined deck environments, MageZero can develop competitive playstyles and meaningful policy/value representations without requiring LLM-scale resources.

MageZero is built to exploit biases of MTG that are often overlooked when discussing MTG AI. (MTG is often seen as having almost no bias and being more of a language problem than a tradition RL game). 
These biases are:  
- **Combinatorial nature:** MTG decks are built around overlapping combinations of synergistic cards. In every deck every card has purpose, and that purpose largely remains the same across different opponents and gamestates. This bias is exploited through mini deck subsets that are part of a training curriculum.  
- **Deck-specific Structure:** While on a whole MTG is as abstract as a language, within the context of one deck MTG is more strucutred than chess. The challenge is identifying and exploiting this structure as it is not explicitly defined. This is why MageZero leans toward lighter less generalizable models, with more dynamic and adaptable feature extraction and learning curriculums.  

#### Core Components:
- **Deck-local learning**: Each agent is trained from scratch on a single deck.
- **Combinatorial minideck bootstrapping**: High-synergy card subsets are automatically identified and simulated in isolation to expose agents to core strategic interactions.
- **Dynamic feature hashing**: A flexible encoding system that supports sparse, open-ended state representation while maintaining fixed-size inputs.
- **Modular architecture**: At any point, training operates on a maximum of two 60 card decks in context: its own and opponents.

---

### 2. Inspirations

MageZero draws from a range of research traditions:
- **AlphaZero (Silver et al., 2017)**: Self-play with Monte Carlo Tree Search and joint policy/value networks.
- **Feature hashing (Weinberger et al., 2009)**: Sparse encodings for large-scale discrete input spaces.
- **Curriculum learning (Bengio et al., 2009)**: Gradual exposure to increasing task complexity.

while there is no one source this project borrows heavily from current trends in meta learning and dynamic curriculums.

### 3. Pipeline Overview
 - Xmage simulations: this is the main hardware bottleneck for me right now, each simulation runs on a seperate thread and my cpu from 2017 only has 8 cores; in addition to this as simulations are built in java, the heap space fills up rapidly (I only have 16gb RAM). This bottleneck has prevented me from being able to start training and testing the model since currently a 20 turn mcts simulation on a medium difficulty deck takes 15-20min based on testing and can often fail due to heap errors. 
 - State Encoding/Vectorization: This system is implemented and tetsted within Xmage. As game states are encountered during self play, a hierarchal map of features is hashed in real time, this multilayered map structure is stored in Features.java. This map structure is serialized to a file and loaded before each simulation to keep mappings consistent between sims. Each vector created from each state is labeled with Win/loss and MCTS derived policy and exported for training in pytorch. This is as much as ive implemented so far.
 - DL models: standard lightweight feedforward network. Since I havent been able to test/work on this yet, optimizer, layers, loss function, etc are all TBD. The goal/design of MageZero is to keep this model as lightweight as possible and ideally alleviate as much learning stress as possible through engineered bias/redundency in the feature vector, and synergy based training curriculum. AlphaZero used a convulation layer to introduce bias, for MageZero minidecks fill that role.
 - Minideck Curriculum: the planned solution for finding implicit structure within decks. The goal of this curriculum will be to identify the synergies unique to each deck from the ground up. This is the core design of MageZero and what I will spend the most time researching. Nothing is set in stone at the moment since I still need to get the model working, but the plan is to use a 'Hasse-walk' approach on a Hasse diagram to stochastically sample minideck combinations based on previously identified synergy.
- Largescale Simulation: the final step/goal. Start creating a rich 60 card simulation envirement where agents learn to master different matchups in addition to their own deck. 

### 4. Game Engine & Feature Encoding

MageZero is implemented atop XMage, an open-source MTG simulator, with custom integration via `ComputerPlayer8.java`.

GitHub Repository: [https://github.com/WillWroble/mage](https://github.com/WillWroble/mage)

Game state is captured via `StateEncoder.java`, which converts each decision point into a binary feature vector reflecting:
- Card presence by zone
- Active abilities and attachments (within each card)
- unique multi-occurence hashes for each feature
- game meta data (life total, cards per zone, etc)
- numeric features are represented as discrete levels (life total 20,19,18 all map to seperate bits) cardinality is represented through thresholds (the bit for 20 life turns on every life total bit below it so it effectively resprsents the condition of having 20 or more life)
- subfeatures (abilities within specific creature) are mapped uniquely per parent but also pool up to parent for additional abstraction (llanawar elves on the battlefield has the subfeature: 'green' and will pass this up to the battlefield feature which will start a count of all green permanents on the battlefield which passes to all zones etc.)

Features are dynamically assigned to slots in a preallocated bit vector (e.g., 20,000 bits) on first occurrence. Completely redundent co-occuring features can be masked without disrupting index alignment. during testing this has resulted in a final binary vector of around 2500 features after 100 simulated games with heurisitc mcts. (2000 with heuristic minimax)

---

### 5. Neural Network Architecture

`MageZero uses a lightweight feedforward network with dual heads:
- **Policy head**: Trained on MCTS visit counts
- **Value head**: Trained on game outcome estimates

The network input is a unified feature vector with partitions for base game, deck-specific and opponent-specific features. tentative planned input size is 4000 bits based on observed encoding data. (because features are hashed dynamically extra space is required for features discovered during traing)

- Initialization: Base is pre initialized (this includes features like life total, turn phase etc.) player and opponent weights are zero or small random initialized.
- Optimization: Standard AlphaZero loss for now (cross-entropy + MSE).

---

### 6. Training Flow

- Pure MCTS rollout on singleton and pair minidecks for feature discovery
- Synergy-based minideck curriculum construction
- Full-deck fine-tuning via self-play
- Matchup tuning using opponent-specific modules

---
### 7. Theoretical Hasse scheduler

- Initially simulates minideck evirements for each individual card and pair of cards, creating a Hasse diagram
- Calculate synergy metric between single cards by measuring difference in combined winrate of cards vs individual winrates.
- Use this synergy metric to stochastically sample set expansions from the leaf nodes of the Hasse diagram for further sims. the goal is to create groups of minidecks that have strong intersynergy with each other. This is to train the agent in a structured way from the ground up in high signal low noise envirements. This is necessary for mtg where many decks have a very rigid and contrived play style ie. combo decks.

---
### 7. Current Progress

- Dynamic encoder and feature merging system implemented and tested in Java
- Custom MCTS variant integrated into XMage environment
- Training infrastructure prepared but currently bottlenecked by hardware limitations

---

### 8. Core Challenges

- **State space sparsity**: MTG's enormous and irregular board state space demands a non-static representation.
- **No pre-learned representation or embedding**: Since MageZero's goal is to avoid generalizing to all of MTG, older simpler vectorization schemes for massive discrete spaces must be used.   
- **Subset explosion**: Efficient sampling and pruning strategies are necessary to keep the synergy discovery tractable.
- **Hardware bottlenecks**: MCTS simulations remain expensive, limiting the scale of training and evaluation.
- **Evaluation methodology**: No gold standard exists for MTG AI benchmarking. Internal metrics and simulated leagues are in development.
- **Long-horizon reward discovery**: Some interactions (e.g., Millennium Calendar Probem) only emerge after dozens of turns. Mini-deck isolation is critical for providing early reward gradients.

---

### 9. Future Directions

- Build a competent mono-green agent as a baseline.
- Experiment with hand crafted minideck curriculums for mono green agent to see how it affects the learning process.
- Expand hand made mini-deck curriculum to more complex archetypes (elves, devotion, combo)
- Study tradeoffs between engineered and emergent features
- Start working on theoretical Hasse model for automated curriculum.
- Develop a reusable library of matchup-specific opponent modules
- Explore curriculum scheduling heuristics and auto-adjustment
- Prepare research publication on dynamic hashing and combinatorial 'minideck' RL
- Build a public-facing simulation engine for scalable deck diagnostics

---

MageZero treats Magic not as one enormous problem, but as a modular ecosystem of tractable subgames. By letting agents specialize, synergize, and grow within structured but flexible boundaries, it opens the door to structured learning in one of the most strategically rich environments ever designed.

