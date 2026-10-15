"""paper_common.py -- text blocks shared by all ten papers (parameterised, not copy-pasted numbers)."""
import json, os, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "out")
PAP = os.path.normpath(os.path.join(_HERE, "..", "paper"))

REFS = {
 "sb": "Sutton, R. S. and Barto, A. G. Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.",
 "bellman": "Bellman, R. Dynamic Programming. Princeton University Press, 1957.",
 "puterman": "Puterman, M. L. Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley, 1994.",
 "watkins": "Watkins, C. J. C. H. and Dayan, P. Q-learning. Machine Learning 8(3-4):279-292, 1992.",
 "singh96": "Singh, S. P. and Sutton, R. S. Reinforcement learning with replacing eligibility traces. Machine Learning 22:123-158, 1996.",
 "komorowski": "Komorowski, M., Celi, L. A., Badawi, O., Gordon, A. C. and Faisal, A. A. The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. Nature Medicine 24:1716-1720, 2018.",
 "raghu": "Raghu, A., Komorowski, M., Celi, L. A., Szolovits, P. and Ghassemi, M. Continuous state-space models for optimal sepsis treatment: a deep reinforcement learning approach. Machine Learning for Healthcare (MLHC), 2017.",
 "gottesman": "Gottesman, O. et al. Guidelines for reinforcement learning in healthcare. Nature Medicine 25:16-18, 2019.",
 "yu": "Yu, C., Liu, J., Nemati, S. and Yin, G. Reinforcement learning in healthcare: a survey. ACM Computing Surveys 55(1):1-36, 2021.",
 "iyengar": "Iyengar, G. N. Robust dynamic programming. Mathematics of Operations Research 30(2):257-280, 2005.",
 "nilim": "Nilim, A. and El Ghaoui, L. Robust control of Markov decision processes with uncertain transition matrices. Operations Research 53(5):780-798, 2005.",
 "weissman": "Weissman, T., Ordentlich, E., Seroussi, G., Verdu, S. and Weinberger, M. Inequalities for the L1 deviation of the empirical distribution. Hewlett-Packard Labs Technical Report HPL-2003-97, 2003.",
 "altman": "Altman, E. Constrained Markov Decision Processes. Chapman and Hall/CRC, 1999.",
 "bradtke": "Bradtke, S. J. and Duff, M. O. Reinforcement learning methods for continuous-time Markov decision problems. Advances in Neural Information Processing Systems 7, 1995.",
 "sutton99": "Sutton, R. S., Precup, D. and Singh, S. Between MDPs and semi-MDPs: a framework for temporal abstraction in reinforcement learning. Artificial Intelligence 112(1-2):181-211, 1999.",
 "levine": "Levine, S., Kumar, A., Tucker, G. and Fu, J. Offline reinforcement learning: tutorial, review, and perspectives on open problems. arXiv:2005.01643, 2020.",
 "jin": "Jin, Y., Yang, Z. and Wang, Z. Is pessimism provably efficient for offline RL? Proceedings of the 38th International Conference on Machine Learning (ICML), 2021.",
 "kumar": "Kumar, A., Zhou, A., Tucker, G. and Levine, S. Conservative Q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems 33, 2020.",
 "fujimoto": "Fujimoto, S., Meger, D. and Precup, D. Off-policy deep reinforcement learning without exploration. Proceedings of the 36th ICML, 2019.",
 "garcia": "Garcia, J. and Fernandez, F. A comprehensive survey on safe reinforcement learning. Journal of Machine Learning Research 16:1437-1480, 2015.",
 "shield": "Alshiekh, M., Bloem, R., Ehlers, R., Konighofer, B., Niekum, S. and Topcu, U. Safe reinforcement learning via shielding. Proceedings of the AAAI Conference on Artificial Intelligence, 2018.",
 "murphy03": "Murphy, S. A. Optimal dynamic treatment regimes. Journal of the Royal Statistical Society: Series B 65(2):331-366, 2003.",
 "murphy05": "Murphy, S. A. A generalization error for Q-learning. Journal of Machine Learning Research 6:1073-1097, 2005.",
 "nahum": "Nahum-Shani, I. et al. Just-in-time adaptive interventions (JITAIs) in mobile health: key components and design principles for ongoing health behavior support. Annals of Behavioral Medicine 52(6):446-462, 2018.",
 "yomtov": "Yom-Tov, E. et al. Encouraging physical activity in patients with diabetes: intervention using a reinforcement learning system. Journal of Medical Internet Research 19(10):e338, 2017.",
 "gupta": "Gupta, D. and Denton, B. Appointment scheduling in health care: challenges and opportunities. IIE Transactions 40(9):800-819, 2008.",
 "zacharias": "Zacharias, C. and Pinedo, M. Appointment scheduling with no-shows and overbooking. Production and Operations Management 23(5):788-801, 2014.",
 "maillart": "Maillart, L. M., Ivy, J. S., Ransom, S. and Diehl, K. Assessing dynamic breast cancer screening policies. Operations Research 56(6):1411-1427, 2008.",
 "ayer": "Ayer, T., Alagoz, O. and Stout, N. K. A POMDP approach to personalize mammography screening decisions. Operations Research 60(5):1019-1034, 2012.",
 "kansagara": "Kansagara, D. et al. Risk prediction models for hospital readmission: a systematic review. JAMA 306(15):1688-1698, 2011.",
 "singer": "Singer, M. et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 315(8):801-810, 2016.",
 "reyna": "Reyna, M. A. et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine 48(2):210-217, 2020.",
 "kdigo": "KDIGO CKD Work Group. KDIGO 2012 clinical practice guideline for the evaluation and management of chronic kidney disease. Kidney International Supplements 3(1):1-150, 2013.",
 "whelton": "Whelton, P. K. et al. 2017 ACC/AHA guideline for the prevention, detection, evaluation, and management of high blood pressure in adults. Hypertension 71(6):e13-e115, 2018.",
 "winstein": "Winstein, C. J. et al. Guidelines for adult stroke rehabilitation and recovery. Stroke 47(6):e98-e169, 2016.",
 "hovorka": "Hovorka, R. et al. Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes. Physiological Measurement 25(4):905-920, 2004.",
 "fox": "Fox, I., Lee, J., Pop-Busui, R. and Wiens, J. Deep reinforcement learning for closed-loop blood glucose control. Proceedings of Machine Learning for Healthcare (PMLR 126), 2020.",
 "efron": "Efron, B. and Morris, C. Stein's paradox in statistics. Scientific American 236(5):119-127, 1977.",
 "gelman": "Gelman, A. and Hill, J. Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press, 2007.",
 "singh00": "Singh, S., Jaakkola, T., Littman, M. L. and Szepesvari, C. Convergence results for single-step on-policy reinforcement-learning algorithms. Machine Learning 38(3):287-308, 2000.",
 "evendar": "Even-Dar, E. and Mansour, Y. Learning rates for Q-learning. Journal of Machine Learning Research 5:1-25, 2003.",
 "rmax": "Brafman, R. I. and Tennenholtz, M. R-max - a general polynomial time algorithm for near-optimal reinforcement learning. Journal of Machine Learning Research 3:213-231, 2002.",
 "mohri": "Mohri, M., Rostamizadeh, A. and Talwalkar, A. Foundations of Machine Learning. MIT Press, 2012.",
 "alpaydin": "Alpaydin, E. Introduction to Machine Learning, 3rd ed. MIT Press, 2014.",
 "schaefer": "Schaefer, A. J., Bailey, M. D., Shechter, S. M. and Roberts, M. S. Modeling medical treatment using Markov decision processes. In Operations Research and Health Care, Kluwer, 2005.",
 "bergman": "Bergman, R. N., Ider, Y. Z., Bowden, C. R. and Cobelli, C. Quantitative estimation of insulin sensitivity. American Journal of Physiology 236(6):E667-E677, 1979.",
 "kovatchev": "Kovatchev, B. P., Breton, M., Dalla Man, C. and Cobelli, C. In silico preclinical trials: a proof of concept in closed-loop control of type 1 diabetes. Journal of Diabetes Science and Technology 3(1):44-55, 2009.",
 "hoeffding": "Hoeffding, W. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association 58(301):13-30, 1963.",
 "ng99": "Ng, A. Y., Harada, D. and Russell, S. Policy invariance under reward transformations: theory and application to reward shaping. Proceedings of the 16th International Conference on Machine Learning (ICML), 1999.",
}


def load(idea):
    with open(f"{OUT}/idea{idea:02d}/results.json") as f:
        return json.load(f)


def fig(idea, name):
    return f"{OUT}/idea{idea:02d}/{name}.png"


def pm(x, ci, nd=2):
    return f"{x:.{nd}f} ± {ci:.{nd}f}"


def pstr(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ---------------------------------------------------------------------------
def background(P, notes=""):
    P.h1("Preliminaries: Markov decision processes and the three solution families")
    P.p("A finite Markov decision process (MDP) is a tuple (S, A, P, R, γ) with a finite state set S, a finite action set A, a "
        "transition kernel P(s' | s, a), an expected reward R(s, a) and a discount factor γ in [0, 1). A deterministic policy π maps "
        "each state to an action. Its value V^π(s) is the expected discounted return obtained by starting in s and following π; the "
        "optimal value V*(s) satisfies the Bellman optimality equation [@sb,bellman,puterman]")
    P.eq("V*(s) = max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s') ]", "1")
    P.p("**Dynamic programming (DP).** When P and R are known, V* is the unique fixed point of the contraction in Eq. (1). Value "
        "iteration repeatedly applies the right-hand side until the sup-norm change is below a tolerance; policy iteration alternates "
        "exact policy evaluation (solving the linear system (I − γP_π)V = R_π) with greedy improvement and terminates in a finite number "
        "of steps. We implement both and verify on every run that they return policies of identical value, which acts as a built-in "
        "correctness test of the solver.")
    P.p("**Monte Carlo (MC) control.** Without a model, MC control estimates the action-value Q(s,a) as the average return observed after "
        "first visits to (s,a) within complete episodes, and improves an ε-soft policy that is greedy with respect to Q [@sb,singh96]. It makes no "
        "bootstrapping assumption, so it is unbiased but has high variance and must wait until an episode ends before learning anything.")
    P.p("**Q-learning.** Tabular Q-learning is a model-free temporal-difference method that bootstraps from its own estimates [@watkins]:")
    P.eq("Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') − Q(s,a) ]", "2")
    P.p("It converges to Q* if every (s,a) is visited infinitely often and the step sizes satisfy the usual stochastic-approximation "
        "conditions. In practice the speed at which it spreads reward information backwards is its main advantage over MC, and its "
        "bias from bootstrapping off noisy estimates is its main weakness.")
    P.p("**Empirical MDPs from logged data.** Clinical data are logged, not simulated: we observe transitions (s, a, r, s') generated by "
        "clinicians. Counts N(s,a,s') give the maximum-likelihood model P̂(s'|s,a) = N(s,a,s')/N(s,a) and R̂(s,a) as the mean observed reward. "
        "Solving the MDP (P̂, R̂) by DP gives a plug-in policy; MC and Q-learning can be run against samples drawn from (P̂, R̂), which is "
        "how \"learning from experience\" is realised without touching patients. Two safeguards are used throughout because plug-in "
        "planning is notoriously exploited by estimation error [@levine,jin,kumar]: (i) a **support constraint**: a pair (s,a) observed fewer than "
        "10 times is completed pessimistically (it transitions to an absorbing terminal state with the worst reward in the data), and "
        "(ii) a **terminal pseudo-count**: one pseudo-observation of absorption is added to every supported pair so that no state is "
        "estimated to be immortal. " + notes)


def protocol(P, rows, steps, seeds, n_eval, extra=""):
    P.h1("Experimental protocol")
    P.h2("Simulators that mirror the named Kaggle datasets")
    P.p("**Important disclosure.** The Kaggle website could not be reached from the environment in which this study was built (all requests "
        "returned HTTP 403), so no Kaggle file was downloaded. Instead every experiment is run on a transparent, hand-specified stochastic "
        "simulator whose recorded variables mirror the columns of the dataset named in the paper. The simulator plays two roles: it "
        "generates a logged dataset by running an ε-noisy clinician-like behaviour policy (the stand-in for the Kaggle table), and it is the "
        "ground truth against which every learned policy is scored. Using a simulator is a methodological choice with a real benefit "
        "(true policy value is measurable, which is impossible with a static table) and a real cost (the numbers are not clinical "
        "evidence). The accompanying code includes a data adapter that converts a real Kaggle CSV into the same transition format so that "
        "the identical pipeline can be re-run on the real file; the results below should be read as a method study, not as a finding about "
        "patients.")
    P.h2("Data, solvers and scoring")
    P.table(rows, "Experimental configuration.", widths=[5.0, 11.0])
    P.bullets([
        f"**Learners.** MC control and Q-learning each receive {steps:,} environment steps drawn from the estimated MDP, with an exponentially "
        "decaying ε-greedy schedule (ε from 1.0 to about 0.05) and a visit-count step size α = max(0.02, 1/(1 + n^0.7)) for Q-learning.",
        f"**Replicates.** {seeds} independent data draws (each with its own logged dataset and learner seeds). Intervals are 95% t-style "
        "intervals over replicates; paired differences are tested with a one-sample t-test on the per-replicate differences.",
        f"**Scoring.** Each final policy is run for {n_eval} episodes in the true simulator using common random numbers (episode e of every policy "
        "starts from the same random stream), and the mean discounted return is recorded. Learning curves are scored exactly on a large "
        "'oracle' MDP estimated from ten or more times as much logged data.",
        "**Fallback.** In states never observed in the (supported) data a learned policy defers to the clinician's modal logged action, "
        "so that unsupported states cannot be exploited.",
    ])
    if extra: P.p(extra)


def limitations(P, specific):
    P.h1("Limitations and threats to validity")
    P.bullets(specific + [
        "**Simulated, not clinical, evidence.** All numbers come from stylised simulators written for this study. They demonstrate how the "
        "algorithms behave when the ground truth is known; they say nothing about the safety or benefit of any treatment in people.",
        "**State aggregation.** Continuous physiology is discretised into bands. Banding makes the process only approximately Markov, and part of "
        "the gap between the DP policy on the estimated model and the true optimum is a modelling gap rather than a sampling gap.",
        "**Fixed-horizon episodes treated as stationary.** Episodes end after a fixed number of steps but the state does not always contain the "
        "clock, so the estimated model treats termination as a constant hazard. This distorts absolute model values (not true-simulator "
        "scores) and is why model-based value estimates are compared only on a common oracle model.",
        "**Behaviour policy.** The logged data come from a single ε-noisy rule. Coverage of non-rule actions is therefore thin, which is exactly "
        "the offline-RL situation, and results depend on that coverage.",
        "**Hand-set hyper-parameters.** ε schedules, step sizes and the support threshold were fixed in advance and not tuned per paper.",
    ])


def defense(P, qas):
    P.h1("Appendix A. Anticipated reviewer questions")
    P.p("The following questions are the ones a reviewer or examiner is most likely to raise. Each answer is tied to a number or figure produced by the code.")
    for i, (q, a) in enumerate(qas, 1):
        P.p(f"**Q{i}. {q}**")
        P.p(a, indent=True)


def repro(P, script, figs):
    P.h1("Appendix B. Reproducibility")
    P.p(f"All results in this paper are produced by `python ideas.py {script}` (shared library: rl_core.py, domains.py, common.py). Random "
        "seeds are fixed in the code, so a re-run reproduces every table and figure. Figure files generated by the script: " + ", ".join(figs) + ". "
        "To repeat the study on a real Kaggle table, convert it with kaggle_adapter.transitions_from_dataframe, then call build_mdp, "
        "value_iteration, mc_control and q_learning exactly as the script does; because a real table has no simulator, report model-based "
        "estimates on a held-out split and label them as estimates.")


ORDER = ["Behavior", "Rule", "DP", "Monte Carlo", "Q-learning", "Oracle DP"]
NICE = {"Behavior": "Clinician behaviour (modal logged action)", "Rule": "Guideline rule", "DP": "DP on estimated MDP",
        "Monte Carlo": "Monte Carlo control", "Q-learning": "Q-learning", "Oracle DP": "Oracle DP (10x data)"}


def std_rows(summ, info_keys=(), info_labels=(), nd=2, order=ORDER, ind=None):
    rows = [["Method", "Return (mean ± 95% CI)"] + list(info_labels)]
    for k in order:
        v = summ[k]
        rows.append([NICE.get(k, k), pm(v["J"], v["ci"], nd)] + [f"{v['info'][i]:.{nd}f}" for i in info_keys])
    return rows


def learning_sentence(res_extra):
    """one sentence on seeds / agreement"""
    ag = res_extra["agree"]
    return (f"Averaged over replicates, Q-learning agreed with the DP policy in {100 * np.mean(ag['Q-learning']):.0f}% of well-visited states "
            f"and Monte Carlo in {100 * np.mean(ag['Monte Carlo']):.0f}%.")


def code_appendix(P, idea_no, extra_funcs=()):
    """Appendix C: the exact source used for this paper (domain class + experiment function)."""
    import inspect, domains, ideas
    P.h1("Appendix C. Source code of the experiment")
    P.p("For completeness the exact source of the simulator and the experiment function is reproduced below; it is the same code that produced every number and figure in this paper "
        "(the shared solvers value_iteration, mc_control and q_learning are in rl_core.py and are delivered with the code package).")
    for obj in list(extra_funcs):
        P.h3(getattr(obj, "__name__", "code"))
        P.code(inspect.getsource(obj))
    P.h3(f"ideas.idea{idea_no}")
    P.code(inspect.getsource(getattr(ideas, f"idea{idea_no}")))


def ethics_future(P, future):
    P.h1("Ethical considerations and future work")
    P.p("**Ethics.** No human data were used, so no consent or privacy issues arise in this study. If the method were applied to real records, three safeguards would be essential: "
        "de-identification and governance approval of the source data; a clinician-in-the-loop design in which the learned policy is advisory, with hard guardrails on unsafe actions; "
        "and prospective, randomised or at least carefully monitored evaluation before any deployment. Reinforcement learning from logs inherits the biases of the clinicians who "
        "produced them, and may under-serve groups that are under-represented in the data; subgroup performance should be reported alongside averages.")
    P.p("**Future work.** " + future)


def rl_core_appendix(P):
    import inspect, rl_core
    P.h3("rl_core: shared solvers")
    for fn in (rl_core.build_mdp, rl_core.value_iteration, rl_core.q_learning, rl_core.mc_control):
        P.code(inspect.getsource(fn))
