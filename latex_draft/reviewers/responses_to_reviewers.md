Response to Reviewers
Manuscript [TBD: journal manuscript ID]: Vaccination Policy under Model Uncertainty: Can the Needs of the Few Outweigh the Needs of the Many?

Operational tracker: [responses_checklist.md](responses_checklist.md). Reviewer reports verbatim: [reviewers_comments.md](reviewers_comments.md).

\===== Reviewer \#1 \=====

\--- Overall comment \---

Reviewer:
This is a very good paper which proposes an alternative model for infectious disease spread under conditions of uncertainty. The proposed model yields conclusions and supports recommendations substantially different from those normally derived from models, especially those used during the pandemic. By proposing a plausible model (that is, one based on plausible, albeit questionable, assumptions about disease spread, human behaviour, virulence changes etc) and suggesting that selective, rather than population-wide, vaccination could result in fewer deaths given certain assumptions, the paper emphasises the uncertainty and speculative nature of many epidemiological models used for policy recommendation, thus making a case for (what I take to be) much needed epistemic humility in model-based epidemiology and subsequent policy recommendations.

The paper is clearly written and well structured and I think it would make a nice contribution to the literature on pandemic/epidemic preparedness and on model-based policy making.

For these reasons, I recommend that it is published, though I have some suggestions for improvements and I should point out that my expertise is in ethics/philosophy only. I cannot therefore assess the technical aspects of the proposed model.

Response:
[TBD — Phase 1: thank the reviewer; signpost the structure of what follows. Distinguish what is adopted as a revision (C1, C2, C6 and, in part, C3–C5), what is defended with reasoning, and what is deferred.]

\--- Comment 1: Engage the selective-lockdown ethics literature \---

Reviewer:
About the ethics of selective vaccination, the authors rightly say that additional normative premises are needed to draw ethical implications and policy recommendation from a model predicting fewer deaths with selective vaccination. The literature on selective lockdowns, which were defended ethically on the basis of similar considerations (creating immunity among the non-vulnerable by letting them free to live their normal lives while temporarily shielding the elderly and vulnerable) might offer some insights that the authors might want to engage with or at least refer to. See e.g. Savulescu J, Cameron J Why lockdown of the elderly is not ageist and why levelling down equality is wrong Journal of Medical Ethics 2020;46:717-721.

Response:
[TBD — Phase 1. Note: the structural parallel the reviewer draws (shield the vulnerable, let immunity build among the non-vulnerable) is close to this paper's Strategy 2, so this is a natural addition to §4 "Limits of Ethical Inference from the Model". Decide whether to engage substantively or to refer.]

Concretely, [TBD — Phase 3: section references plus before/after quotations.]

\--- Comment 2: Engage the modelling-humility and uncertainty-communication literature \---

Reviewer:
About the uncertainty around models, their susceptibility to contested assumption, and the importance of epistemic humility that these factors, many of the recommendations put forward by the authors in the conclusion align with the recommendations in Saltelli A, et al 2020. Five ways to ensure that models serve society: a manifesto. Nature. Jun;582(7813):482-484. The authors might want to engage with or refer to those. And about the importance of communicating uncertainty around public health policy, the authors might want to see Giubilini, A et al 2025, "Expertise, Disagreement, and Trust in Vaccine Science and Policy. The Importance of Transparency in a World of Experts". Diametros 22 (82): 7-27.

Response:
[TBD — Phase 1. Both references land naturally in §6 Conclusions; Saltelli et al. also bears on the §2 framing that positions the paper within the philosophy of models in science and policy.]

Concretely, [TBD — Phase 3.]

\--- Comment 3: Ground the confidence claim in the Simplifications section \---

Reviewer:
The section on Simplifications of the model might benefit from further clarifications. In particular, the authors write at the end of that section that "Despite all these simplifications, we are confident that the main qualitative conclusions our model licences can also be drawn from modified models and/or evaluation metrics". Where does that confidence come from? Why would the reader trust your confidence? Surely, appeal to your own sense of confidence by itself does not provide sufficient support.

Response:
[TBD — Phase 1. This is a fair hit: the sentence asserts robustness without argument. Options span (a) replacing the assertion with the actual grounds — the mechanism is driven by viral ageing plus differential exposure, which is not an artefact of the grid or the movement rule; (b) pointing to the robustness evidence already in the paper (the parameter-space optimisation of §5, the worst-case analysis of §5.5); (c) weakening the claim to a conditional. Decide the mix in Phase 1.]

Concretely, [TBD — Phase 3.]

\--- Comment 4: Clarify what is negative about "negative herd immunity" \---

Reviewer:
The notion of 'negative herd immunity" the authors introduce at p. 14 might need to be better explained. The authors write "This situation can be understood as a sort of negative herd immunity effect, where herd immunity may have negative consequences for those who remain susceptible". It's not very clear why the implications are negative. I suppose herd immunity is good in any case for those susceptible as, by definition, it means a significantly reduced risk of infection (compared to a situation of non-herd immunity). It is not clear to me that 'herd immunity' itself is what is negative, as opposed to the way in which it is achieved. If I understand well, on this model, vaccine-induced herd immunity would be less good than collective immunity obtained by allowing natural infection among the non-vulnerable, as the latter is more likely than the former to reduce virulence. But that simply means that the latter is preferable to the former (though neither is negative) and that what is more or less preferable is the way in which herd immunity is achieved, rather than the outcome itself.

Response:
[TBD — Phase 1. The reviewer's reconstruction ("the route matters, not the outcome") is largely right and can be conceded; the question is whether to retain the term "negative herd immunity" (which is borrowed from Luyten et al. 2011, where it names a different mechanism — age-shifted severity) or to retire it in favour of a formulation about the route to collective immunity. Note the interaction with C5: whichever notion of herd immunity is adopted has to be consistent here.]

Concretely, [TBD — Phase 3.]

\--- Comment 5: Specify which notion of herd immunity is in use \---

Reviewer:
Moreover, 'herd immunity' can mean different things (from a dynamic equilibrium that allows for periodical waves of the virus, as per Sunetra Gupta's understanding, to simply very low risk that a susceptible and an infectious person come into contact). It would be good to clarify which understanding of herd immunity is used here, as that seems relevant to an epidemilogical assessment of the model.

Response:
[TBD — Phase 1. Requires stating which notion the SISD model actually realises. Note that the model has no permanent removed compartment — recovered agents return to susceptible with accumulated immunity — so the operative notion is closer to a dynamic equilibrium than to a threshold-crossing one. Confirm against `src/simulation_class.py` before committing to a formulation.]

Concretely, [TBD — Phase 3.]

\--- Comment 6: Introduce block quotations properly \---

Reviewer:
Small comment: There are citations that are added to the text without context or being introduced properly. For instance it is unclear where the citation at pp.5-6 comes from and how it relates to the discussion there, and same for citation at p. 8. Citations should be properly introduced in a way that makes them fit with the flow of the text.

Response:
[TBD — Phase 1. Straightforwardly adopted. Requires first confirming which block quotations the page references point to — see the checklist, where three candidates are listed.]

Concretely, [TBD — Phase 3.]

\===== Reviewer \#2 \=====

\--- Overall comment \---

Reviewer:
Thank you for asking me to review this paper. The more general points in the paper about the limits of modelling and of assumptions that everyone must be vaccinated are important points. However, the main thrust of the paper depends on a doubtful empirical claim, that pandemic viruses decline in virulence.

Response:
[TBD — Phase 1: acknowledge the concession on the general points, then signpost the reply to the central objection below.]

\--- Comment 1: The virulence-decline premise \---

Reviewer:
However, the main thrust of the paper depends on a doubtful empirical claim, that pandemic viruses decline in virulence. While many people might believe this claim, and while there might be evidence for effects of this kind, such effects are clearly dwarfed by the effects (acting in the same direction on average disease severity) of acquired immunity, whether via vaccination or (as was more common in historic pandemics) previous infection.

Consider these counter-examples: (1) Hong Kong & omicron - a very high death rate was observed due to a lack of immunity due to low prior infection rates and low vaccination among older adults; in other words, low population immunity was a much bigger factor than any decline in the intrinsic virulence of omicron. (2) remote communities of the pacific or the arctic where epidemics of 1918 flu were typically late but nonetheless severe. More generally (3) the high mortality from smallpox and measles when introduced to immuno-naive populations via colonisation, suggesting again that a lack of prior immunity was a much bigger factor than decline in virulence from centuries of endemic transmission in Europe.

These difficulties with the core assumption make it difficult to evaluate any ethical implications that follow.

Response:
[TBD — Phase 1. This is the load-bearing objection and needs the most care. Points available for the framing, to be selected and sharpened with the user:

- The paper's argumentative role is modal, not predictive — it is a constructive counterexample to the claim that mass vaccination *must* reduce deaths, so the premise needs to be *possible and not unreasonable*, not *typical*. §4 (`subsec:EthicalLimits`) already says this; the reply may need to make it load-bearing rather than a caveat.
- The reviewer's counterexamples establish that population immunity dominates *in those cases*; they do not establish that the transmission–virulence trade-off is absent. The model implements a mechanism (Kun 2023, Duffy 2018), and the model also *includes* acquired immunity as a separate channel, so the two effects are not conflated in the code.
- Whether the model's comparative result survives when the immunity channel is strengthened relative to the viral-ageing channel is, in principle, checkable in this codebase (`src/simulation_class.py`; the parameter sweep in `01_optimization/`). Decide with the user whether to run that robustness check — it would be the strongest possible reply, and it is the one item here that may require new simulation work rather than prose.
- Honest concession: the size of the virulence-decline effect in real pandemics is contested, and the paper should say so rather than lean on it.

Decide in Phase 1 whether this is answered by reframing alone, by reframing plus new simulations, or by both.]

Concretely, [TBD — Phase 3.]

\--- Comment 2: The tetanus claim in the introduction \---

Reviewer:
The introduction also contains the incorrect claim that tetanus has been virtually eradicated via herd immunity - this is not true, at least not insofar as "herd immunity" means something like reduced population transmission arising from the presence of transmission-reducing immunity in large numbers of individuals, because tetanus is not transmitted between people.

Response:
[TBD — Phase 1. The reviewer is correct: tetanus is acquired from environmental *C. tetani* spores and is not transmitted person to person, so it cannot be controlled by herd immunity. Straightforwardly adopted; the fix is local to the opening sentence. Note the interaction with C5 and R1·C4 — whatever definition of herd immunity is settled on should be the one this sentence is corrected against.]

Concretely, [TBD — Phase 3.]

\===== Paper Changes Checklist \=====

[TBD — Phase 2/3: mirror the operational checklist once the Phase 1 framings are approved and the atomic actions are fixed. Keep in sync with responses_checklist.md.]

\[R1 · C1\] Engage the selective-lockdown ethics literature
• [TBD]

\[R1 · C2\] Engage the modelling-humility and uncertainty-communication literature
• [TBD]

\[R1 · C3\] Ground the confidence claim in the Simplifications section
• [TBD]

\[R1 · C4\] Clarify what is negative about "negative herd immunity"
• [TBD]

\[R1 · C5\] Specify which notion of herd immunity is in use
• [TBD]

\[R1 · C6\] Introduce block quotations properly
• [TBD]

\[R2 · C1\] The virulence-decline premise
• [TBD]

\[R2 · C2\] The tetanus claim in the introduction
• [TBD]
