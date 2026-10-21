// build_slides.js -- builds the Topic 7 presentation from slide_numbers.json and the figures of run_real.py.
//
//     python topic7/presentation/slide_numbers.py     (numbers from results.json)
//     node topic7/presentation/build_slides.js        (needs node and pptxgenjs: npm install pptxgenjs)
//
// Writes topic7/presentation/Topic7_Overbooking_Presentation.pptx: a navy title slide, white content slides with a
// numbered section label, serif headings, tinted panels, navy table headers and the paper's own figures. Every point
// is at most three lines long. Speaker notes on every slide say who presents it and what to say.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const N = JSON.parse(fs.readFileSync(path.join(HERE, "slide_numbers.json"), "utf8"));
const FIG = (name) => path.join(HERE, "..", "code", "out", "topic7_real", `${name}.png`);
const OUTFILE = path.join(HERE, "Topic7_Overbooking_Presentation.pptx");

// ---------------------------------------------------------------- look
const HEAD = "Cambria", BODY = "Calibri";
const C = { navy: "1B2A4A", ink: "1F2933", muted: "5B6770", tint: "EEF2F8", line: "C9D2DE", gold: "A97C2F",
  goldTint: "F8EEDA", ice: "CADCFC", zebra: "F7F9FC", white: "FFFFFF" };
const W = 13.333, X0 = 0.6, CW = W - 2 * X0, TOP = 1.55;
const PRAKRIT = "Speaker: Prakrit.", NAGASAI = "Speaker: Nagasai.";
const SEC = { intro: "01   INTRODUCTION", data: "02   DATA AND METHOD", results: "03   RESULTS",
  wrap: "04   IMPLEMENTATION AND DISCUSSION", refs: "REFERENCES", backup: "BACKUP" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "M Naga Sai Dattu, Prakrit Bahl";
pres.title = "Overbooking Clinic Appointments with Reinforcement Learning";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

const footer = { text: { text: "Overbooking Clinic Appointments with Reinforcement Learning   |   Topic 7", options: {
  x: X0, y: 7.0, w: 9, h: 0.3, fontFace: BODY, fontSize: 10, color: C.muted, margin: 0 } } };
const slideNum = { x: W - X0 - 0.8, y: 7.0, w: 0.8, h: 0.3, fontFace: BODY, fontSize: 10, color: C.muted, align: "right" };
const sectionPh = { placeholder: { options: { name: "section", type: "body", x: X0, y: 0.32, w: 9, h: 0.3, fontFace: BODY,
  fontSize: 12, bold: true, color: C.gold, charSpacing: 1.5, margin: 0, valign: "middle" }, text: "" } };
const titlePh = { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.62, w: CW, h: 0.75, fontFace: HEAD,
  fontSize: 28, bold: true, color: C.navy, valign: "middle", align: "left", margin: 0 }, text: "" } };

pres.defineSlideMaster({ title: "TITLE_SLIDE", background: { color: C.navy }, objects: [] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: C.white }, slideNumber: slideNum, objects: [
  sectionPh, titlePh, footer,
  { placeholder: { options: { name: "body", type: "body", x: X0, y: TOP, w: CW, h: 5.25, fontFace: BODY, fontSize: 18,
    color: C.ink, valign: "top", margin: 0 }, text: "" } },
] });
pres.defineSlideMaster({ title: "TITLE_ONLY", background: { color: C.white }, slideNumber: slideNum, objects: [sectionPh, titlePh, footer] });

// ---------------------------------------------------------------- helpers
// a point is a string, or {b: "bold lead", t: " rest"}; numbered lists pass num = true
function runs(items, size = 18, gap = 9, num = false) {
  const out = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const parts = typeof it === "string" ? [{ text: it }] : [{ text: it.b, bold: true }, { text: it.t || "" }];
    parts.forEach((p, j) => {
      const o = { bold: !!p.bold, fontSize: size, color: C.ink, fontFace: BODY };
      if (j === 0) Object.assign(o, { bullet: num ? { type: "number", indent: 22 } : { indent: 18 }, paraSpaceAfter: gap });
      if (j === parts.length - 1 && !last) o.breakLine = true;
      out.push({ text: p.text, options: o });
    });
  });
  return out;
}
function start(master, section, title, notes) {
  const sec = section === SEC.refs ? "IMPLEMENTATION AND DISCUSSION" : section.split("   ").pop();
  const s = pres.addSlide({ masterName: master, sectionTitle: sec });
  s.addText(section, { placeholder: "section" });
  s.addText(title, { placeholder: "title" });
  s.addNotes(notes);
  return s;
}
function content(section, title, items, notes, size = 18) {
  const s = start("CONTENT", section, title, notes);
  s.addText(runs(items, size, 10), { placeholder: "body" });
  return s;
}
function text(s, items, x, y, w, h, size = 16, opts = {}) {
  s.addText(runs(items, size, opts.gap || 8, opts.num), { x, y, w, h, valign: "top", margin: 0, isTextBox: true, objectName: opts.name || "text" });
}
function panel(s, x, y, w, h, heading, items, size = 16, opts = {}) {
  // a tinted rounded panel with a serif heading and points inside
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: opts.fill || C.tint }, line: { color: opts.fill || C.tint },
    rectRadius: 0.08, objectName: `${opts.name || "panel"}_bg` });
  let ty = y + 0.2;
  if (heading) {
    s.addText(heading, { x: x + 0.25, y: ty, w: w - 0.5, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.navy, margin: 0,
      isTextBox: true, objectName: `${opts.name || "panel"}_head` });
    ty += 0.5;
  }
  if (items) text(s, items, x + 0.25, ty, w - 0.5, h - (ty - y) - 0.15, size, { ...opts, name: `${opts.name || "panel"}_text` });
}
function stat(s, x, y, w, h, value, label, name, vsize = 36) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.tint }, line: { color: C.tint }, rectRadius: 0.08, objectName: `${name}_bg` });
  const vh = vsize / 48;
  s.addText(value, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: vh, fontFace: HEAD, fontSize: vsize, bold: true, color: C.navy, margin: 0,
    isTextBox: true, objectName: `${name}_value` });
  s.addText(label, { x: x + 0.25, y: y + 0.2 + vh, w: w - 0.5, h: h - 0.3 - vh, fontFace: BODY, fontSize: 14, color: C.muted, valign: "top",
    margin: 0, isTextBox: true, objectName: `${name}_label` });
}
function table(s, rows, x, y, colW, opts = {}) {
  // first row is the header (navy); opts.hl: row indexes to highlight (gold tint, bold); zebra body rows
  const size = opts.size || 14, hl = new Set(opts.hl || []);
  const data = rows.map((r, i) => r.map((cell, j) => ({
    text: cell,
    options: { bold: i === 0 || hl.has(i), fontSize: size, fontFace: BODY, color: i === 0 ? C.white : C.ink, valign: "middle",
      align: j === 0 || (opts.leftCols || []).includes(j) ? "left" : "center",
      fill: { color: i === 0 ? C.navy : hl.has(i) ? C.goldTint : i % 2 ? C.white : C.zebra } },
  })));
  s.addTable(data, { x, y, colW, w: colW.reduce((a, b) => a + b, 0), border: { type: "solid", pt: 0.5, color: C.line },
    margin: [4, 7, 4, 7], rowH: opts.rowH || 0.36, objectName: opts.name || "table" });
}
function figure(s, name, x, y, w, aspect, altText) {
  // the paper's figure on a white card with a thin border; altText also keeps the local path out of the file
  const h = w / aspect;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x - 0.1, y: y - 0.1, w: w + 0.2, h: h + 0.2, fill: { color: C.white },
    line: { color: C.line, width: 0.75 }, rectRadius: 0.06, objectName: `${name}_frame` });
  s.addImage({ path: FIG(name), x, y, w, h, objectName: name, altText });
}

// ================================================================ 1 title
{
  pres.addSection({ title: "INTRODUCTION" });
  const s = pres.addSlide({ masterName: "TITLE_SLIDE", sectionTitle: "INTRODUCTION" });
  s.addText("TOPIC 7   |   APPOINTMENT OVERBOOKING AND NO-SHOW MANAGEMENT", { x: X0, y: 1.0, w: CW, h: 0.35, fontFace: BODY, fontSize: 13,
    bold: true, color: C.ice, charSpacing: 1.5, margin: 0, isTextBox: true, objectName: "topic" });
  s.addText("Overbooking Clinic Appointments with Reinforcement Learning", { x: X0, y: 1.55, w: CW, h: 1.5, fontFace: HEAD, fontSize: 40,
    bold: true, color: C.white, valign: "top", margin: 0, isTextBox: true, objectName: "title" });
  s.addText("Exploration Schedules, Learning Efficiency and Robustness to Shifts in the No-Show Rate", { x: X0, y: 3.1, w: CW, h: 0.5,
    fontFace: HEAD, fontSize: 20, italic: true, color: C.ice, margin: 0, isTextBox: true, objectName: "subtitle" });
  s.addText([
    { text: "M Naga Sai Dattu (23BCE0757)", options: { bold: true, breakLine: true } },
    { text: "Prakrit Bahl (23BCE0826)", options: { bold: true, breakLine: true } },
    { text: "School of Computer Science and Engineering (SCOPE)", options: { breakLine: true } },
    { text: "Vellore Institute of Technology, Vellore" },
  ], { x: X0, y: 4.6, w: CW, h: 1.6, fontFace: BODY, fontSize: 17, color: C.white, margin: 0, paraSpaceAfter: 4, isTextBox: true, objectName: "authors" });
  s.addNotes(`${PRAKRIT} Introduce both presenters and the topic. This project takes Paper 7 and its code, evaluates all of its methods on the real Kaggle data, and proposes a method that outperforms them. Prakrit presents the problem, the data and the methods (slides 1 to 8); Nagasai presents the proposed method, the results and the implementation (slides 9 to 18).`);
}

// ================================================================ 2 problem
{
  const s = start("TITLE_ONLY", SEC.intro, "Problem Context: Patient No-Shows and Overbooking",
    `${PRAKRIT} Explain the balance a clinic has to strike. About one in five patients does not attend. Booking additional patients fills those slots, but if most patients attend, the clinic works overtime. The decision recurs every session and affects the waiting list, which is why it is modelled as a Markov decision process and addressed with reinforcement learning.`);
  text(s, [
    `Patient no-shows leave scheduled clinic capacity unused; in the dataset studied here, ${N.data.noshow} of appointments were missed.`,
    "Clinics therefore overbook, issuing more appointments than there are regular slots in a session.",
    "Booking too few additional patients leaves slots idle and lengthens the waiting list, whereas booking too many causes overtime when most patients attend.",
    "The appropriate number of additional bookings depends on the patients scheduled, and therefore varies from one session to the next.",
    "Because each decision also affects the waiting list for later sessions, the task is a sequential decision problem and is addressed with reinforcement learning (RL) [3].",
  ], X0, TOP, 8.0, 5.3, 18, { gap: 12 });
  stat(s, 9.15, TOP, CW - 8.55, 2.2, N.data.noshow, `of the ${N.data.appointments} recorded appointments were missed`, "noshow");
  stat(s, 9.15, TOP + 2.45, CW - 8.55, 2.2, "0 to 4", "additional patients that the clinic may book before each session", "actions");
}

// ================================================================ 3 problem statement
{
  const s = start("TITLE_ONLY", SEC.intro, "Problem Statement and How It Is Addressed",
    `${PRAKRIT} Go through the problem statement line by line. Every requirement is met on the real data: the state uses the risk band predicted by the classifier, as required; the action is the overbooking level; the reward is Paper 7's reward; and all four evaluation measures are reported on slides 10 to 14.`);
  table(s, [
    ["Requirement", "How this work addresses it"],
    ["Dataset: Medical Appointment No Shows", `All reported results are computed on the public Kaggle table of ${N.data.appointments} appointments recorded in 2016 [17].`],
    ["State: slots filled and predicted no-show risk band", `Slots filled (3 bands) × risk band predicted by a calibrated classifier (4) × waiting list band (5), giving ${N.mdp.states} states.`],
    ["Action: overbooking level", `Between 0 and 4 additional patients are booked before each session (${N.mdp.actions} actions).`],
    ["Reward: utilisation less waiting and overtime", "shows − 3 × overtime − 0.5 × idle slots − 0.4 × waiting list + 0.3 × patients admitted from the waiting list, as in Paper 7."],
    ["Futuristic angle: adaptation to patient behaviour", "The policy responds to the predicted risk of each session and to the waiting list, and is evaluated under shifts in the no-show rate."],
    ["Evaluation: utilisation, waiting, overtime, convergence of MC and Q-learning", "All four measures are reported on scoring weeks never used for learning (slides 10 to 14)."],
  ], X0, TOP, [4.3, CW - 4.3], { size: 15, rowH: 0.7, leftCols: [1], name: "statement" });
}

// ================================================================ 4 paper 7
{
  const s = start("TITLE_ONLY", SEC.intro, "Paper 7: The Baseline Study",
    `${PRAKRIT} Paper 7 was provided with its code. It used a simulator because the Kaggle site could not be reached when it was written. Its learned policies improved on the rule by only ${N.paper7_gain.DP} to ${N.paper7_gain["Monte Carlo"]}, while DP with ten times the data improved by ${N.paper7_gain["Oracle DP"]}; the learners were therefore limited by the data rather than by the algorithm. Our objective follows from this.`);
  panel(s, X0, TOP, 7.3, 3.45, "Scope of the original study", [
    "Formulated overbooking as a Markov decision process (MDP) with 100 states and 5 actions.",
    "Evaluated all methods on a simulator specified by hand that reproduces the columns of the dataset, as the Kaggle site could not be accessed at the time.",
    `Learned policies from ${N.mdp.days} logged clinic days using dynamic programming (DP), Monte Carlo (MC) control and Q-learning.`,
    "Compared five exploration schedules and examined robustness to a change in the no-show rate.",
  ], 15, { name: "scope" });
  panel(s, 8.15, TOP, CW - 7.55, 3.45, "Reported returns (simulator)", null, 15, { name: "found" });
  table(s, [
    ["Method", "Return"],
    ["Guideline rule", N.paper7.Rule],
    ["DP on estimated MDP", N.paper7.DP],
    ["Monte Carlo control", N.paper7["Monte Carlo"]],
    ["Q-learning", N.paper7["Q-learning"]],
    ["Oracle DP (10x data)", N.paper7["Oracle DP"]],
  ], 8.4, TOP + 0.7, [2.9, 1.18], { size: 14, rowH: 0.42, name: "paper7" });
  panel(s, X0, TOP + 3.7, CW, 1.6, null, [
    { b: "Identified gap: ", t: `the learned policies improved on the guideline rule by only ${N.paper7_gain.DP} to ${N.paper7_gain["Monte Carlo"]}, whereas DP with ten times as much data improved on it by ${N.paper7_gain["Oracle DP"]}; no real data were used.` },
    { b: "Objective of this work: ", t: "to evaluate Paper 7's methods on the real dataset, to propose a method that outperforms them, and to report both in the same paper." },
  ], 16, { name: "gap", fill: C.goldTint });
}

// ================================================================ 5 data
{
  pres.addSection({ title: "DATA AND METHOD" });
  const s = start("TITLE_ONLY", SEC.data, "The Medical Appointment No Shows Dataset",
    `${PRAKRIT} Describe the dataset and the main driver of no-shows. The classifier assigns each booked patient a probability of not attending; the state of the MDP uses the risk band of each session. The classifier is trained only on earlier weeks, so no information from the scoring weeks is used.`);
  panel(s, X0, TOP, 5.9, 5.25, "Dataset characteristics", [
    `${N.data.appointments} appointments of ${N.data.patients} patients across ${N.data.neighbourhoods} neighbourhoods in Brazil, recorded on ${N.data.days} weekdays between 29 April and 8 June 2016 [17].`,
    `The overall no-show rate is ${N.data.noshow}, and it varies little by weekday (${N.data.weekday_lo} to ${N.data.weekday_hi}).`,
    `Booking lead time is the strongest predictor: ${N.data.lead_same} of same day bookings were missed, compared with ${N.data.lead_31} of bookings made 31 or more days in advance.`,
    "Before the analysis, one record with a negative age and the 39 appointments held on Saturdays were removed.",
  ], 16, { name: "dataset", gap: 12 });
  panel(s, 6.75, TOP, CW - 6.15, 5.25, "No-show classifier", [
    "Gradient boosting [18], calibrated by isotonic regression [19] and implemented in scikit-learn [20].",
    "Uses only information known at booking: lead time, age, sex, SMS reminder, health and welfare flags, weekday, history and neighbourhood.",
    "Trained on appointments before 9 May, calibrated on 9 to 15 May and applied from 16 May onwards, so no later information is used.",
    `Test AUC of ${N.data.auc}; the mean predicted no-show rate is ${N.data.pred} against ${N.data.obs} observed.`,
  ], 16, { name: "classifier", gap: 12 });
}

// ================================================================ 6 MDP
{
  const s = start("TITLE_ONLY", SEC.data, "Formulating the Problem as a Markov Decision Process",
    `${PRAKRIT} Explain how the table becomes an MDP. Every policy is learned on the May sessions and evaluated on the June sessions, which it has never seen; this is what allows true performance to be measured on real data.`);
  const cells = [
    ["Session", `All appointments of one neighbourhood on one day (at least five), with ${N.mdp.slots} regular slots.`],
    ["Episode", `${N.mdp.sessions} sessions drawn from the real sessions while preserving the weekday order.`],
    ["State", `Slots filled (5 to 7, 8 to 9, 10) × predicted risk quartile (4) × waiting list band (5) = ${N.mdp.states} states.`],
    ["Action", "The number of additional patients booked before the session: 0, 1, 2, 3 or 4."],
    ["Reward", "shows − 3 × overtime − 0.5 × idle − 0.4 × waiting list + 0.3 × patients admitted from the waiting list."],
    ["Data split", `Learning: 16 to 25 May (${N.data.sessions_past} sessions). Scoring: 30 May to 8 June (${N.data.sessions_future} sessions), unseen in learning.`],
  ];
  const pw = (CW - 0.6) / 3, ph = 2.05;
  cells.forEach(([h, t], i) => {
    const x = X0 + (i % 3) * (pw + 0.3), y = TOP + Math.floor(i / 3) * (ph + 0.25);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: pw, h: ph, fill: { color: C.tint }, line: { color: C.tint }, rectRadius: 0.08, objectName: `mdp${i}_bg` });
    s.addText(h, { x: x + 0.25, y: y + 0.2, w: pw - 0.5, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.navy, margin: 0, isTextBox: true, objectName: `mdp${i}_head` });
    s.addText(t, { x: x + 0.25, y: y + 0.7, w: pw - 0.5, h: ph - 0.85, fontFace: BODY, fontSize: 16, color: C.ink, valign: "top", margin: 0, isTextBox: true, objectName: `mdp${i}_text` });
  });
  s.addText(`Attendance of every booked patient is taken directly from the records; policies are learned from ${N.mdp.days} logged clinic days, as in Paper 7.`,
    { x: X0, y: TOP + 2 * ph + 0.45, w: CW, h: 0.45, fontFace: BODY, fontSize: 16, italic: true, color: C.muted, margin: 0, isTextBox: true, objectName: "mdp_note" });
}

// ================================================================ 7 gaps
{
  const s = start("TITLE_ONLY", SEC.data, "Limitations of the Data and How They Are Addressed",
    `${PRAKRIT} Present this slide openly; it is the first question an examiner is likely to ask. Only attendance is real data. The decisions in the logs are generated with Paper 7's own behaviour policy, and the waiting list follows Paper 7's rule because the table contains none. The paper states the same in Section 6.1 and in its limitations.`);
  table(s, [
    ["Not recorded in the dataset", "Approach adopted"],
    ["Overbooking decisions", "Logs are generated by applying Paper 7's behaviour policy to the real sessions; attendance remains as recorded."],
    ["Waiting list", "Paper 7's waiting list rule is retained (1.5 new patients per session); this is the only simulated component."],
    ["Clinics and slots", `A session is defined as one neighbourhood on one day, with ${N.mdp.slots} regular slots.`],
    ["Times within a day", "Overtime and waiting are measured in patients, as in Paper 7."],
    ["More than 26 days of data", `${N.mdp.days} logged days of ${N.mdp.sessions} sessions are sampled from the ${N.data.sessions_past} learning sessions.`],
    ["Variation in the no-show rate", `Recorded outcomes of randomly selected patients are reversed, moving the rate from ${N.rates[0]} to ${N.rates[1]} or ${N.rates[2]}.`],
  ], X0, TOP, [3.7, CW - 3.7], { size: 16, rowH: 0.68, leftCols: [1], name: "gaps" });
}

// ================================================================ 8 methods
{
  const s = start("TITLE_ONLY", SEC.data, "Methods Compared",
    `${PRAKRIT} Paper 7's three learners estimate the full transition model from the logs. The published baselines are simplified implementations, and their settings were tuned on the scoring weeks, which favours them. Every method is scored on the same ${N.mdp.episodes} episodes with the same random numbers. Hand over to Nagasai for the proposed method.`);
  panel(s, X0, TOP, 5.9, 3.55, "Methods from Paper 7", [
    "DP (value iteration) on the MDP estimated from the logged data.",
    "Monte Carlo control and Q-learning on the same estimated MDP.",
    "The guideline rule (one or two additional bookings) and the clinician behaviour policy.",
    `Oracle DP: DP fitted to ${N.mdp.oracle_days} logged days, ten times the standard amount, as a reference.`,
  ], 16, { name: "paper7methods" });
  panel(s, 6.75, TOP, CW - 6.15, 3.55, "Baselines from published work", [
    "Single session optimiser: chooses the number of additional bookings that maximises the expected reward of the current session [6].",
    "Threshold double booking (θ = 0.25): one additional booking for each patient whose predicted risk is at least θ [7].",
    "Fixed booking: always books k = 2 additional patients.",
  ], 16, { name: "baselines" });
  panel(s, X0, TOP + 3.8, CW, 1.45, "Evaluation protocol", [
    `Each method is evaluated over ${N.mdp.seeds} replicates (independent logged datasets) and ${N.mdp.episodes} episodes of the scoring sessions with common random numbers; the return is the discounted reward over ${N.mdp.sessions} sessions.`,
  ], 16, { name: "protocol", fill: C.goldTint });
}

// ================================================================ 9 our method
{
  const s = start("TITLE_ONLY", SEC.data, "Proposed Method: A Structured Model of the Clinic",
    `${NAGASAI} Paper 7's learners estimate every transition separately, so the actions that matter most (booking three or four additional patients) are rarely observed. Most of the clinic, however, is known. We learn only patient behaviour, ${N.mdp.n_est} parameters informed by every logged session, and derive everything else from the clinic's rules. This is a factored MDP, and MC and Q-learning can also be trained on it.`);
  panel(s, X0, TOP, 5.9, 3.6, "Motivation", [
    `Paper 7's methods estimate every transition probability P(s' | s, a) from counts: ${N.mdp.n_emp} quantities from ${N.mdp.days} days.`,
    "Booking three or four additional patients is rare in the logs, so the estimates for these actions are unreliable.",
    "Most of the clinic's dynamics are known in advance: the slots, the reward and how additional bookings reduce the waiting list.",
  ], 16, { name: "motivation" });
  panel(s, 6.75, TOP, CW - 6.15, 3.6, "Procedure", [
    "Estimate only patient behaviour from the logs: cell frequencies (12), the waiting list arrival rate (1) and waiting list lengths within each band (13).",
    "Compute the expected reward of each session exactly from the patients' predicted attendance probabilities.",
    "Assemble the transition model from these components, a factored MDP [13].",
    "Solve it by value iteration (DP); MC and Q-learning can also be trained on it.",
  ], 16, { name: "procedure", num: true });
  stat(s, X0, TOP + 3.85, 3.0, 1.4, `${N.mdp.n_est}`, "parameters in the structured model", "n_est", 30);
  stat(s, X0 + 3.25, TOP + 3.85, 3.0, 1.4, N.mdp.n_emp, "transition probabilities in the estimated MDP", "n_emp", 30);
  text(s, ["Every logged session informs all of the structured model's parameters, which is why it needs far less data than the estimated MDP."],
    X0 + 6.55, TOP + 4.0, CW - 6.55, 1.2, 16, { name: "why" });
}

// ================================================================ 10 main result
{
  pres.addSection({ title: "RESULTS" });
  const s = start("TITLE_ONLY", SEC.results, "Main Results on the Real Scoring Sessions",
    `${NAGASAI} Read the table from the top: the guideline rule, the published baselines, Paper 7's learners and the oracle, then the structured model. It attains ${N.real["Ours: DP"]}: ${N.ours_vs.DP.d} over DP, ${N.ours_vs["Q-learning"].d} over Q-learning and ${N.ours_vs["Monte Carlo"].d} over MC (all ${N.ours_vs.DP.p}), and ${N.ours_vs_ep["Oracle DP"].d} over the oracle with ten times the data. Its margin over the single session optimiser is small (${N.ours_vs_ep["Single session optimiser"].d}) but consistent; slide 13 shows when it grows. It also has the shortest waiting list.`);
  const rows = [["Method", "Return (± 95% CI)", "Utilisation", "Overtime", "Waiting list"]];
  const hl = [];
  N.table.forEach((r, i) => { rows.push([(r.paper7 ? "Paper 7: " : "") + r.name, r.J, r.util, r.over, r.wait]); if (r.ours && r.name.includes("DP")) hl.push(i + 1); });
  table(s, rows, X0, TOP - 0.05, [3.85, 1.75, 1.15, 1.0, 1.15], { size: 13, rowH: 0.36, hl, name: "results" });
  s.addText("Return: discounted reward over 40 sessions (mean ± 95% CI over 20 replicates). Overtime and waiting list are per session.",
    { x: X0, y: 6.45, w: 8.9, h: 0.4, fontFace: BODY, fontSize: 12, italic: true, color: C.muted, margin: 0, isTextBox: true, objectName: "caption" });
  const sx = 9.85, sw = W - X0 - sx;
  stat(s, sx, TOP - 0.05, sw, 1.62, N.real["Ours: DP"], "return of the structured model, the highest of all methods", "best", 32);
  stat(s, sx, TOP + 1.72, sw, 1.62, N.ours_vs.DP.d, `over DP, the best of Paper 7's methods (${N.ours_vs.DP.p})`, "vs_dp", 32);
  stat(s, sx, TOP + 3.49, sw, 1.62, N.ours_vs_ep["Single session optimiser"].d, `over the strongest published baseline (${N.ours_vs_ep["Single session optimiser"].p})`, "vs_single", 32);
}

// ================================================================ 11 data size
{
  const s = start("TITLE_ONLY", SEC.results, "Effect of the Amount of Logged Data",
    `${NAGASAI} This slide explains most clearly why the proposed method performs better. DP on the estimated MDP needs hundreds of days to approach its best value, and with 25 days it falls below the rule. The structured model attains ${N.size[0].ours} with 25 days because it estimates ${N.mdp.n_est} parameters that every session informs.`);
  figure(s, "fig_datasize", X0 + 0.1, TOP + 0.1, 7.0, 1011 / 644, "Return on the real scoring sessions against the number of logged clinic days, for DP on the estimated MDP and for the structured model");
  const last = N.size[N.size.length - 1];
  text(s, [
    `With ${N.size[0].days} logged days, DP on the estimated MDP attains ${N.size[0].dp.split(" ")[0]}, below the guideline rule (${N.real.Rule}); with ${last.days} days it reaches ${last.dp.split(" ")[0]}.`,
    `The structured model attains ${N.size[0].ours} at every data size from ${N.size[0].days} to ${last.days} days.`,
    `The reason is the number of quantities estimated: ${N.mdp.n_est} parameters instead of ${N.mdp.n_emp} transition probabilities.`,
    "The limitation of Paper 7's methods is therefore the coverage of the logged data, not the learning algorithm.",
  ], 8.1, TOP, CW - 7.5, 5.3, 16, { gap: 12 });
}

// ================================================================ 12 shift
{
  const s = start("TITLE_ONLY", SEC.results, "Robustness to Changes in the No-Show Rate",
    `${NAGASAI} The scoring sessions are rescored after reversing recorded outcomes, so the no-show rate rises to ${N.rates[1]} or falls to ${N.rates[2]}. The policies are not retrained. An increase in no-shows affects the rule most; the structured model remains ahead of every method at all three rates.`);
  const rows = [["Method", `Nominal rate (${N.rates[0]})`, `Higher rate (${N.rates[1]})`, `Lower rate (${N.rates[2]})`]];
  const hl = [];
  N.shift.forEach((r, i) => { rows.push([r.name, ...r.vals]); if (r.ours) hl.push(i + 1); });
  table(s, rows, X0, TOP, [3.0, 1.5, 1.5, 1.5], { size: 15, rowH: 0.5, hl, name: "shift" });
  panel(s, 8.45, TOP, W - X0 - 8.45, 4.4, "Observations", [
    "The structured model has the highest return at all three rates.",
    `Rising no-shows affect the guideline rule most: its return falls from ${N.shift[0].vals[0]} to ${N.shift[0].vals[1]}.`,
    "Policies are not retrained; only the recorded attendance outcomes are altered.",
  ], 16, { name: "obs", gap: 12 });
}

// ================================================================ 13 demand
{
  const s = start("TITLE_ONLY", SEC.results, "Benefit of Planning Across Sessions Under Higher Demand",
    `${NAGASAI} The single session optimiser considers only the current session. When waiting is more costly or more patients join the list, planning across sessions pays off, and the gain grows from ${N.demand[0].gain} to ${N.demand[N.demand.length - 1].gain}. DP on the estimated MDP deteriorates in the busiest setting because the actions it needs are rarely logged.`);
  const rows = [["Waiting cost", "New patients per session", "DP on estimated MDP", "Best fixed booking", "Single session optimiser", "Structured model", "Gain"]];
  N.demand.forEach((r) => rows.push([r.cost, r.arrivals, r.dp, r.fixed, r.single, r.ours, r.gain]));
  table(s, rows, X0, TOP, [1.5, 1.95, 1.95, 1.75, 2.05, 1.75, 1.18], { size: 15, rowH: 0.5, name: "demand" });
  text(s, [
    "The single session optimiser considers only the current session, whereas the structured model also accounts for the effect of each decision on the waiting list in later sessions.",
    `This advantage grows with demand: the gain over the single session optimiser rises from ${N.demand[0].gain} to ${N.demand[N.demand.length - 1].gain}.`,
    `DP on the estimated MDP deteriorates sharply in the busiest setting (${N.demand[N.demand.length - 1].dp}), because the actions it requires are rarely observed in the logs.`,
    "A waiting cost of 0.4 with 1.5 new patients per session corresponds to Paper 7's setting (slide 10).",
  ], X0, TOP + 3.0, CW, 2.3, 16, { gap: 9 });
}

// ================================================================ 14 convergence
{
  const s = start("TITLE_ONLY", SEC.results, "Convergence of Monte Carlo Control and Q-learning",
    `${NAGASAI} The problem statement asks for this comparison. The curves show the true return of the greedy policy as learning proceeds. Q-learning ends higher than MC on both models, and both perform better on the structured model. In Paper 7's exploration schedule study, rerun on the real data, no schedule comes within 90% of the optimum, so exploration is not the bottleneck; the data are.`);
  figure(s, "fig_convergence", X0 + 0.1, TOP + 0.1, 6.5, 1178 / 762, "True return of the greedy policy during learning for Monte Carlo and Q-learning, on the estimated MDP and on the structured model");
  text(s, [
    `On the estimated MDP, MC converges to a return of ${N.conv["Monte Carlo|sir"].final} and Q-learning to ${N.conv["Q-learning|sir"].final}.`,
    `On the structured model, Q-learning reaches ${N.conv["Q-learning|ours"].final} and MC ${N.conv["Monte Carlo|ours"].final}, against ${N.real["Ours: DP"]} for its DP solution.`,
    "Q-learning approaches the DP solution more closely than MC on both models.",
    `No exploration schedule reached 90% of the possible improvement in any of ${N.sched.runs} runs.`,
    `For Q-learning, near greedy exploration was the slowest schedule (AUC ${N.sched.auc_ng} against ${N.sched.auc_best}).`,
    `The ranking of the schedules differs between MC and Q-learning (rank correlation ${N.sched.rho}).`,
  ], 7.55, TOP, W - X0 - 7.55, 5.3, 16, { gap: 10 });
}

// ================================================================ 15 code
{
  pres.addSection({ title: "IMPLEMENTATION AND DISCUSSION" });
  const s = start("TITLE_ONLY", SEC.wrap, "Implementation and Reproducibility",
    `${NAGASAI} All code is in the public repository, folder topic7. Paper 7's code is reused without modification; our files only import and call it. A single command reruns every experiment, and the paper is generated from the results file, so no number in the paper or on these slides is typed by hand.`);
  panel(s, X0, TOP, 6.6, 4.6, "Software pipeline (Python)", [
    { b: "noshow_data.py: ", t: "data cleaning, classifier training and construction of sessions." },
    { b: "real_clinic.py: ", t: "the clinic MDP built from the real sessions." },
    { b: "planner.py: ", t: "the structured model proposed in this work." },
    { b: "rl_core.py, common.py, ideas.py: ", t: "Paper 7's solvers and experiments, reused without modification." },
    { b: "run_real.py: ", t: "executes all experiments and writes results.json and the figures." },
    { b: "paper07_v3.py, paper07_v4.py: ", t: "generate the paper from results.json; check scripts verify it." },
  ], 15, { name: "pipeline" });
  panel(s, 7.45, TOP, CW - 6.85, 4.6, "Models, tools and reproducibility", [
    "Gradient boosting classifier with isotonic calibration (scikit-learn).",
    "Value iteration, Monte Carlo control and Q-learning from Paper 7's rl_core module.",
    "numpy, scipy, matplotlib and python-docx.",
    { b: "Code: ", t: "github.com/Nagasai-Datta/ml_lab, folder topic7." },
    { b: "Reproduction: ", t: "python topic7/code/run_real.py reruns every experiment with fixed random seeds." },
  ], 15, { name: "tools" });
}

// ================================================================ 16 limitations
content(SEC.wrap, "Limitations", [
  "Only attendance is observed in the data; the overbooking decisions in the logs are generated and the waiting list is simulated.",
  "The structured model assumes that the clinic's rules (slots, reward weights and waiting list dynamics) are known.",
  `Only about ${N.data.not_full} of real sessions have fewer than 10 bookings, so the slots filled component of the state rarely varies.`,
  `The no-show rate drifted from ${N.data.noshow_past} to ${N.data.noshow_future} between periods; the classifier predicts ${N.data.pred} against ${N.data.obs} observed.`,
  "Waiting is measured by the length of the waiting list, as the dataset records no times within a day.",
  "The published baselines are simplified implementations whose settings were tuned on the scoring sessions, which favours them.",
  "The findings concern algorithms rather than clinical practice, and no clinical claim is made.",
], `${NAGASAI} State these plainly. The first matters most: the data record who attended, not what a clinic decided, so the decisions are generated with Paper 7's behaviour policy. Applying the method in practice would require a clinic that can state its own rules.`);

// ================================================================ 17 conclusion
{
  const s = start("TITLE_ONLY", SEC.wrap, "Conclusions and Future Work",
    `${NAGASAI} Summarise in three sentences: Paper 7's methods work on real data but are limited by the logs; learning only patient behaviour removes that limitation; and the advantage grows when the clinic is busier. Then present the future work, thank the audience and invite questions.`);
  panel(s, X0, TOP, CW, 3.0, "Conclusions", [
    `On the real sessions, Paper 7's DP, MC and Q-learning outperform the guideline rule (${N.real.DP}, ${N.real["Monte Carlo"]} and ${N.real["Q-learning"]} against ${N.real.Rule}) but remain below the oracle (${N.real["Oracle DP"]}), which shows that the logged data are the limiting factor.`,
    `Learning only patient behaviour removes this limitation: the structured model attains ${N.real["Ours: DP"]} and remains the best method when the no-show rate rises or falls, with as few as ${N.size[0].days} logged days.`,
    `Its advantage over the strongest published baseline increases from ${N.demand[0].gain} to ${N.demand[N.demand.length - 1].gain} as pressure on the waiting list grows.`,
  ], 16, { name: "conclusions" });
  panel(s, X0, TOP + 3.25, CW, 2.0, "Future work", [
    "Collect logs in which the number of additional bookings is varied deliberately.",
    "Measure overtime and waiting from recorded visit times.",
    "Estimate the reward weights and waiting list dynamics from clinic records rather than assuming them.",
  ], 16, { name: "future", fill: C.goldTint });
}

// ================================================================ 18 references
{
  const s = start("TITLE_ONLY", SEC.refs, "References",
    `${NAGASAI} References are numbered as in the paper. The dataset is [17]; the published baselines are [6] and [7]; the factored MDP idea behind the proposed method is [13]. Thank the audience and invite questions.`);
  const half = Math.ceil(N.refs.length / 2);
  const col = (list) => list.map((r, i) => ({ text: r, options: { fontSize: 12, fontFace: BODY, color: C.ink, paraSpaceAfter: 4, breakLine: i < list.length - 1 } }));
  s.addText(col(N.refs.slice(0, half)), { x: X0, y: TOP - 0.1, w: 5.95, h: 5.4, valign: "top", margin: 0, isTextBox: true, objectName: "refs_left" });
  s.addText(col(N.refs.slice(half)), { x: X0 + 6.18, y: TOP - 0.1, w: 5.95, h: 5.4, valign: "top", margin: 0, isTextBox: true, objectName: "refs_right" });
}

// ================================================================ backup: expected questions
{
  pres.addSection({ title: "BACKUP" });
  const pick = [3, 7, 8, 9, 10, 1];                       // question numbers from the paper's Appendix A (version 3)
  for (let i = 0; i < pick.length; i += 2) {
    const s = start("TITLE_ONLY", SEC.backup, `Anticipated Questions (${i / 2 + 1} of ${pick.length / 2})`,
      "Backup slide for questions. The answers are the same as in Appendix A of version 3 of the paper.");
    [pick[i], pick[i + 1]].forEach((q, j) => {
      const qa = N.qa[q - 1];
      // point to the slides rather than to the paper's own section and table numbers
      const ans = qa.a.replace("which Section 9 lists as a limitation", "which is one of our limitations (slide 16)")
        .replace("(Table 9)", "(slide 13)").replace("(Section 6.1)", "(slide 7)");
      panel(s, X0, TOP + j * 2.7, CW, 2.45, qa.q, null, 16, { name: `q${q}` });
      s.addText(ans, { x: X0 + 0.25, y: TOP + j * 2.7 + 0.7, w: CW - 0.5, h: 1.6, fontFace: BODY, fontSize: 16, color: C.ink, valign: "top",
        margin: 0, isTextBox: true, objectName: `a${q}` });
    });
  }
}

// pptxgenjs writes a second <a:pPr> (with buNone) after the first run of a paragraph that has several runs, which
// removes the bullet of every "bold lead-in + text" item; drop those stray properties before saving.
const JSZip = require(require.resolve("jszip", { paths: [path.dirname(require.resolve("pptxgenjs"))] }));
pres.write({ outputType: "nodebuffer" }).then(async (buf) => {
  const zip = await JSZip.loadAsync(buf);
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    zip.file(name, xml.replace(/(<\/a:r>)<a:pPr\b[^>]*?(?:\/>|>.*?<\/a:pPr>)/g, "$1"));
  }
  fs.writeFileSync(OUTFILE, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log(OUTFILE);
});
