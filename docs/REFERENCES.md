# References

Each entry below was checked against a bibliographic source (arXiv, publisher or
indexing service) on 2026-11-17. Where a venue could not be confirmed, only the confirmed
record is cited.

## Dataset and prognostics

1. A. Saxena, K. Goebel, D. Simon, N. Eklund. "Damage Propagation Modeling for Aircraft
   Engine Run-to-Failure Simulation." *Proc. 1st International Conference on Prognostics and
   Health Management (PHM08)*, Denver, CO, 2008. (Included with the NASA download; defines
   C-MAPSS, the sensors and the PHM08 score.)
2. F. O. Heimes. "Recurrent neural networks for remaining useful life estimation." *2008
   International Conference on Prognostics and Health Management*, pp. 1–6, 2008.
   doi:10.1109/PHM.2008.4711422. (Early RNN approach on the PHM08 challenge data.)
3. X. Li, Q. Ding, J.-Q. Sun. "Remaining useful life estimation in prognostics using deep
   convolution neural networks." *Reliability Engineering & System Safety* 172:1–11, 2018.
   doi:10.1016/j.ress.2017.11.021. (Time-window CNN on normalised raw C-MAPSS sensors.)

## Architectures

4. S. Bai, J. Z. Kolter, V. Koltun. "An Empirical Evaluation of Generic Convolutional and
   Recurrent Networks for Sequence Modeling." arXiv:1803.01271, 2018. (TCN design used here:
   causal dilated residual convolutions.)
5. K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, Y. Bengio.
   "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine
   Translation." arXiv:1406.1078, 2014. (Introduces the gated recurrent unit.)
6. A. Vaswani et al. "Attention Is All You Need." arXiv:1706.03762, 2017. (Transformer encoder.)

## Uncertainty quantification

7. R. Koenker, G. Bassett Jr. "Regression Quantiles." *Econometrica* 46(1):33–50, 1978.
   doi:10.2307/1913643. (Quantile regression / pinball loss.)
8. J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman. "Distribution-Free
   Predictive Inference for Regression." *Journal of the American Statistical Association*
   113(523):1094–1111, 2018. doi:10.1080/01621459.2017.1307116. (Split conformal prediction and
   its finite-sample marginal coverage.)
9. Y. Romano, E. Patterson, E. J. Candès. "Conformalized Quantile Regression." *NeurIPS 2019*,
   pp. 3543–3553. arXiv:1905.03222. (CQR.)
10. R. F. Barber, E. J. Candès, A. Ramdas, R. J. Tibshirani. "Conformal prediction beyond
    exchangeability." *Annals of Statistics* 51(2):816–845, 2023. doi:10.1214/23-AOS2276.
    (Why coverage guarantees weaken under dependence or distribution shift.)
11. A. N. Angelopoulos, S. Bates. "Conformal Prediction: A Gentle Introduction." *Foundations
    and Trends in Machine Learning*, 2023. doi:10.1561/2200000101; arXiv:2107.07511.

No numerical results from these papers are reproduced or compared in AeroMind. Published
C-MAPSS numbers use different protocols (often test-set evaluation with differing caps),
so they are not directly comparable with the validation results reported here.
