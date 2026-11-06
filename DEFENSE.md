# Project notes

I find H1 **inconclusive** and H2 **unsupported** under the frozen rules. Adding F to the richer baseline changes AUC by -0.008816, with its adjusted interval [-0.02427, 0.004163]. I report 21 test-period event clusters and 10793 common forecast anchors.

I use a Kalman filter for the linear state and observation equations. The original unscented transform produces the same moments in this linear case. I define F as the largest singular-value amplification of the fitted local response over the declared horizon.

I do not claim causality or trading performance. My principal limitations are the small number of drawdown events, the single U.S. return source, and the absence of funding, leverage, and liquidity measures. I disclose the analysis chronology in the paper and amendment record.

I record funding, competing interests, AI assistance, code and data availability, licensing, and the disclaimer in the paper's declarations. I give the clean-clone instructions in README.md.
