# citations etc.

- Borghesi, Andrea, et al. "Anomaly Detection using Autoencoders in High Performance Computing Systems." arXiv preprint arXiv:1811.05269 (2018).

    - suggests deep bottleneck with L1 regularization, rather than constrained bottleneck
    - works well for anomaly detection in high-dimension env (166 variables??)
    - simple NN format (input: 166, bottleneck: 10*166/L1 regularization/ReLU, output: 166/linear)

- C. Zhou, R. Paffenroth. "Anomaly Detection with Robust Deep Autoencoders." KDD'17, August 13-17, 2017, Halifax, NS, Canada.

    - link: https://dl.acm.org/citation.cfm?id=3098052
    - adds a set of exceptions to the enforcement of autoencoder loss functions
    - trains autoencoder to reproduce noise without