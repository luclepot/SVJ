# citations etc.

- Borghesi, Andrea, et al. "Anomaly Detection using Autoencoders in High Performance Computing Systems." arXiv preprint arXiv:1811.05269 (2018).

    - suggests deep bottleneck with L1 regularization, rather than constrained bottleneck
    - works well for anomaly detection in high-dimension env (166 variables??)
    - simple NN format (input: 166, bottleneck: 10*166/L1 regularization/ReLU, output: 166/linear)

- C. Zhou, R. Paffenroth. "Anomaly Detection with Robust Deep Autoencoders." KDD'17, August 13-17, 2017, Halifax, NS, Canada.

    - link: https://dl.acm.org/citation.cfm?id=3098052
    - adds a set of exceptions to the enforcement of autoencoder loss functions
    - trains autoencoder to separate what it takes to be "signal" vs. "noise"

- Chen, Jinghui, et al. "Outlier detection with autoencoder ensembles." Proceedings of the 2017 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2017.

    - link: https://epubs.siam.org/doi/abs/10.1137/1.9781611974973.11
    - introduces ensembles of autoencoders for improved outlier detection wrt benchmarks
    - uses "random edge sampling" (randomly deactivate certain edge weights of autoencoders) to ensure variation within ensembles

- T. Gale, et al. "The State of Sparsity in Deep Neural Networks." arXiv preprint arXiv:1902.09574 (2019)

    - link: https://arxiv.org/abs/1902.09574
    - gives an overview of sparsity enforcement in DNNs
    - asserts specifically that simple sparsity enforcement works better than more complex techniques 
        - mag pruning, l0 norm, etc. easier to enforce
