# GitHub real-AIO pilot: combined per-question table

Google AI Overviews fetched through SerpApi (`serpapi_aio`, gl=in, hl=en, desktop, no_cache=true, skip_evidence). Target = https://github.com/ (github.com and its subdomains). Brand terms: "GitHub", "github.com".

- **Questions are reconstructed** from the abridged wording in the paper's Table II; they are not the original question strings.
- One row per question per observation. Rows are never merged across observations (Q11 appears twice).
- Q8 (question text names GitHub) is excluded from mention analysis and was not run.
- Every figure is read from the stored observation; nothing is estimated. Mention counts are over retrieved overviews (`answer` + `answer_no_references`).
- Other GitHub-owned domains checked: github.blog, githubusercontent.com, githubassets.com, github.io, githubstatus.com, githubapp.com, github.dev, ghcr.io, npmjs.com, npmjs.org. None were cited.
- Step 1 and batch A were stored by an earlier fetcher version; the Notes column explains outcomes it classified differently. Stored values are shown as they are.

## Credits per observation

| Observation | Credits counted (stored) | Balance before → after | Balance change | Difference |
|---|---|---|---|---|
| obs_github_pilot_1791297109 | 15 | 207 → 189 | 18 | 3 |
| obs_github_pilot_A_1791298362 | 40 | 189 → 147 | 42 | 2 |
| obs_github_pilot_B_1791300066 | 20 | 147 → 127 | 20 | 0 |
| obs_github_pilot_B2_1791302154 | 6 | 128 → 122 | 6 | 0 |

## Per question

| Observation | Q | Question | Runs | answer | ans. no refs | no_aio | followup_empty | timeout | parse_error | error | Runs with refs | Runs citing github.com | github.com URLs cited (runs) | Other GitHub-owned | Mentioned (string) / retrieved | Mentioned (excl. absence) / retrieved | Credits counted | Credits assumed | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| obs_github_pilot_1791297109 | Q1 | Which are the best cloud platforms for Indian startups to host their code securely? *(reconstructed)* | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | – | – | 2/2 | 2/2 | 5 | – | run 1: stored as error by the pre-fix fetcher; an empty AI Overview follow-up (now followup_empty); its follow-up credit was billed but not counted |
| obs_github_pilot_1791297109 | Q3 | What should a remote enterprise team look for when choosing a code hosting platform? *(reconstructed)* | 3 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | – | – | 0/1 | 0/1 | 6 | – | run 0: stored as parse_error by the pre-fix fetcher; an overview without a references key (now answer_no_references); not in retrieved_overviews; run 2: stored as parse_error by the pre-fix fetcher; an overview without a references key (now answer_no_references); not in retrieved_overviews |
| obs_github_pilot_1791297109 | Q11 | What is the enterprise-tier price per user, in INR, for code hosting platforms? *(reconstructed)* | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 1 | https://github.com/pricing (1 run) | – | 1/1 | 1/1 | 4 | – | run 0: stored as error by the pre-fix fetcher; an empty AI Overview follow-up (now followup_empty); its follow-up credit was billed but not counted; run 1: stored as error by the pre-fix fetcher; an empty AI Overview follow-up (now followup_empty); its follow-up credit was billed but not counted |
| obs_github_pilot_A_1791298362 | Q2 | How do software teams distributed across Indian cities handle version control? *(reconstructed)* | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | – | – | 3/3 | 3/3 | 6 | – | – |
| obs_github_pilot_A_1791298362 | Q4 | Which code hosting platforms are most widely used by Indian software services companies? *(reconstructed)* | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | – | – | 3/3 | 3/3 | 6 | – | – |
| obs_github_pilot_A_1791298362 | Q11 | What is the enterprise-tier price per user, in INR, for code hosting platforms? *(reconstructed)* | 3 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 2 | https://docs.github.com/en/copilot/concepts/billing-and-usage/organizations-and-enterprises/seats-and-billing-cycles (1 run); https://github.com/pricing (2 runs) | – | 2/2 | 2/2 | 7 | – | – |
| obs_github_pilot_A_1791298362 | Q12 | Do code hosting platforms offer startup discounts or credits in India? *(reconstructed)* | 3 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | – | – | 2/2 | 2/2 | 7 | – | – |
| obs_github_pilot_A_1791298362 | Q13 | How does the pricing of a cloud SaaS code hosting plan compare with a self-hosted deployment? *(reconstructed)* | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | – | – | 1/2 | 1/2 | 4 | – | run 1: client-side timeout stored as error with 0 credits by the pre-fix fetcher (now timeout, 1 credit assumed) |
| obs_github_pilot_A_1791298362 | Q14 | Are there hidden costs for storage or CI/CD minutes on code hosting platforms? *(reconstructed)* | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | https://docs.github.com/billing/managing-billing-for-github-actions/about-billing-for-github-actions (2 runs); https://docs.github.com/en/billing/reference/actions-runner-pricing (1 run); https://docs.github.com/enterprise-cloud@latest/codespaces/managing-codespaces-for-your-organization/managing-the-cost-of-github-codespaces-in-your-organization (1 run) | – | 3/3 | 3/3 | 6 | – | – |
| obs_github_pilot_A_1791298362 | Q15 | What payment methods are available for annual enterprise plans in India? *(reconstructed)* | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | – | – | 0/2 | 0/2 | 4 | – | run 1: client-side timeout stored as error with 0 credits by the pre-fix fetcher (now timeout, 1 credit assumed) |
| obs_github_pilot_B_1791300066 | Q5 | How can a startup improve developer productivity using cloud-based development tools? *(reconstructed)* | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | – | – | 2/2 | 2/2 | 4 | 0 | run 2: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored |
| obs_github_pilot_B_1791300066 | Q6 | How do code hosting platforms handle automated CI/CD pipelines? *(reconstructed)* | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 1 | https://awesome-copilot.github.com/instruction/github-actions-ci-cd-best-practices/ (1 run) | – | 1/1 | 1/1 | 2 | 0 | run 0: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored; run 1: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored |
| obs_github_pilot_B_1791300066 | Q7 | How can dependencies be scanned automatically for security vulnerabilities on a code hosting platform? *(reconstructed)* | 3 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | https://docs.github.com/en/enterprise-server@3.20/code-security/concepts/vulnerability-reporting-and-management/vulnerability-exposure (1 run) | – | 1/1 | 1/1 | 5 | 0 | run 1: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored |
| obs_github_pilot_B_1791300066 | Q9 | How well does a code hosting platform integrate with Jira for sprint planning? *(reconstructed)* | 3 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | – | – | 0/0 | 0/0 | 6 | 0 | run 2: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored |
| obs_github_pilot_B_1791300066 | Q10 | What code review features help senior developers mentor junior developers who work remotely? *(reconstructed)* | 3 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | – | – | 1/1 | 1/1 | 3 | 1 | run 1: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored |
| obs_github_pilot_B2_1791302154 | Q6 | How do code hosting platforms handle automated CI/CD pipelines? *(reconstructed)* | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | – | – | 1/1 | 1/1 | 2 | 0 | run 0: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored; run 1: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored |
| obs_github_pilot_B2_1791302154 | Q7 | How can dependencies be scanned automatically for security vulnerabilities on a code hosting platform? *(reconstructed)* | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 1 | https://github.com/Protik49/Security-Auditing-of-Open-Source-Dependencies (1 run) | – | 1/1 | 1/1 | 2 | 0 | run 1: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored; run 2: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored |
| obs_github_pilot_B2_1791302154 | Q9 | How well does a code hosting platform integrate with Jira for sprint planning? *(reconstructed)* | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | – | – | 1/1 | 1/1 | 2 | 0 | run 0: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored; run 1: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored |
| obs_github_pilot_B2_1791302154 | Q10 | What code review features help senior developers mentor junior developers who work remotely? *(reconstructed)* | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | – | – | 0/0 | 0/0 | 0 | – | run 0: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored; run 1: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored; run 2: SerpApi HTTP 503 "couldn't get valid results"; not billed, nothing stored |

No absence sentences were recorded in any retrieved overview.

## Totals (all observations, stored values)

Runs attempted: 57 across 19 question-observation rows (14 distinct questions).

| Outcome | Runs |
|---|---|
| answer | 27 |
| answer_no_references | 2 |
| no_aio | 0 |
| followup_empty | 5 |
| timeout | 1 |
| parse_error | 2 |
| error | 20 |

- Questions with at least one retrieved overview: 14 of 14 (Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q9, Q10, Q11, Q12, Q13, Q14, Q15).
- Questions with at least one run citing a github.com page: 4 of 14 (Q6, Q7, Q11, Q14).
  - Q6: obs_github_pilot_B_1791300066: 1 run(s)
  - Q7: obs_github_pilot_B_1791300066: 1 run(s); obs_github_pilot_B2_1791302154: 1 run(s)
  - Q11: obs_github_pilot_1791297109: 1 run(s); obs_github_pilot_A_1791298362: 2 run(s)
  - Q14: obs_github_pilot_A_1791298362: 3 run(s)

`error` runs by stored error text: SerpApi HTTP 503 (not billed): 15; empty AI Overview follow-up, pre-fix fetcher: 3; client-side timeout, pre-fix fetcher: 2.

Stored stop reasons:
- obs_github_pilot_B2_1791302154: Stopped: 7 of 12 planned runs ended as error, followup_empty or timeout (limit: more than 50%); remaining observations were not run

Balance outside the runs: between obs_github_pilot_B_1791300066 and obs_github_pilot_B2_1791302154 the balance went from 127 to 128, with no searches by these runs in between.
