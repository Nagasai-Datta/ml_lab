// build_slides.js -- builds the Topic 7 presentation from slide_numbers.json and the figures of run_real.py.
//
//     python topic7/presentation/slide_numbers.py     (numbers from results.json)
//     node topic7/presentation/build_slides.js        (needs node and pptxgenjs: npm install pptxgenjs)
//
// Writes topic7/presentation/Topic7_Overbooking_Presentation.pptx. Plain layout on purpose: white slides, bullet
// points, the paper's own figures and tables. Speaker notes on every slide say who presents it and what to say.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const N = JSON.parse(fs.readFileSync(path.join(HERE, "slide_numbers.json"), "utf8"));
const FIG = (name) => path.join(HERE, "..", "code", "out", "topic7_real", `${name}.png`);
const OUTFILE = path.join(HERE, "Topic7_Overbooking_Presentation.pptx");

// ---------------------------------------------------------------- look
const FONT = "Calibri";
const C = { title: "1F3B4D", text: "222222", muted: "5F6B73", line: "C9CED3", head: "EEF1F3", ours: "E2EEEE" };
const W = 13.333, X0 = 0.6, CW = W - 2 * X0;          // wide layout, 0.6" side margins
const FOOT = "Topic 7: Appointment overbooking and no-show management";
const PRAKRIT = "Speaker: Prakrit.", NAGASAI = "Speaker: Nagasai.";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "M Naga Sai Dattu, Prakrit Bahl";
pres.title = "Overbooking Clinic Appointments with Reinforcement Learning";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

const footer = [
  { text: { text: FOOT, options: { x: X0, y: 7.02, w: 9, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, margin: 0 } } },
];
const slideNum = { x: W - X0 - 0.8, y: 7.02, w: 0.8, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, align: "right" };
const titlePh = { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.35, w: CW, h: 0.8, fontFace: FONT, fontSize: 30,
  bold: true, color: C.title, valign: "middle", margin: 0 }, text: "" } };

pres.defineSlideMaster({ title: "TITLE_SLIDE", background: { color: "FFFFFF" }, objects: [] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: "FFFFFF" }, slideNumber: slideNum, objects: [
  titlePh, ...footer,
  { placeholder: { options: { name: "body", type: "body", x: X0, y: 1.35, w: CW, h: 5.45, fontFace: FONT, fontSize: 18, color: C.text,
    valign: "top", margin: 0 }, text: "" } },
] });
pres.defineSlideMaster({ title: "TITLE_ONLY", background: { color: "FFFFFF" }, slideNumber: slideNum, objects: [titlePh, ...footer] });

// ---------------------------------------------------------------- helpers
// a bullet item is a string, or [string, level], or {b: "bold lead", t: " rest", lvl}
function runs(items, size = 18, gap = 7) {
  const out = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    let lvl = 0, parts;
    if (typeof it === "string") parts = [{ text: it }];
    else if (Array.isArray(it)) { parts = [{ text: it[0] }]; lvl = it[1]; }
    else { parts = [{ text: it.b, bold: true }, { text: it.t || "" }]; lvl = it.lvl || 0; }
    parts.forEach((p, j) => {
      const o = { bold: !!p.bold, fontSize: lvl ? size - 2 : size, color: C.text, fontFace: FONT };
      if (j === 0) Object.assign(o, { bullet: { indent: 18 }, indentLevel: lvl, paraSpaceAfter: gap });
      if (j === parts.length - 1 && !last) o.breakLine = true;
      out.push({ text: p.text, options: o });
    });
  });
  return out;
}
function content(sectionTitle, title, items, notes, size = 20) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle });
  s.addText(title, { placeholder: "title" });
  s.addText(runs(items, size, 10), { placeholder: "body" });
  s.addNotes(notes);
  return s;
}
function titleOnly(sectionTitle, title, notes) {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle });
  s.addText(title, { placeholder: "title" });
  s.addNotes(notes);
  return s;
}
function box(s, items, x, y, w, h, size = 18, name = "text") {
  s.addText(runs(items, size), { x, y, w, h, valign: "top", margin: 0, isTextBox: true, objectName: name });
}
function heading(s, text, x, y, w, name = "heading") {
  s.addText(text, { x, y, w, h: 0.4, fontFace: FONT, fontSize: 20, bold: true, color: C.title, margin: 0, isTextBox: true, objectName: name });
}
function table(s, rows, x, y, colW, opts = {}) {
  // rows: array of arrays of strings; first row is the header; opts.bold: indexes of rows to bold and tint
  const size = opts.size || 14, bold = new Set(opts.bold || []);
  const data = rows.map((r, i) => r.map((cell, j) => ({
    text: cell,
    options: { bold: i === 0 || bold.has(i), fontSize: size, fontFace: FONT, color: C.text, valign: "middle",
      align: j === 0 || opts.leftCols?.includes(j) ? "left" : "center",
      fill: { color: i === 0 ? C.head : bold.has(i) ? C.ours : "FFFFFF" } },
  })));
  s.addTable(data, { x, y, colW, w: colW.reduce((a, b) => a + b, 0), border: { type: "solid", pt: 0.75, color: C.line },
    margin: [3, 6, 3, 6], rowH: opts.rowH || 0.36, objectName: opts.name || "table" });
}
function image(s, name, x, y, w, aspect, altText) {
  s.addImage({ path: FIG(name), x, y, w, h: w / aspect, objectName: name, altText });   // altText also keeps the local path out of the file
}

// ---------------------------------------------------------------- 1 title
{
  pres.addSection({ title: "Introduction" });
  const s = pres.addSlide({ masterName: "TITLE_SLIDE", sectionTitle: "Introduction" });
  s.addText("Overbooking Clinic Appointments with Reinforcement Learning", { x: X0, y: 1.0, w: CW, h: 1.4, fontFace: FONT, fontSize: 36,
    bold: true, color: C.title, align: "center", margin: 0, isTextBox: true, objectName: "title" });
  s.addText("Exploration Schedules, Learning Efficiency and Robustness to Shifts in the No-Show Rate", { x: X0, y: 2.6, w: CW, h: 0.6,
    fontFace: FONT, fontSize: 22, color: C.text, align: "center", margin: 0, isTextBox: true, objectName: "subtitle" });
  s.addText("Topic 7: Appointment Overbooking and No-Show Management", { x: X0, y: 3.4, w: CW, h: 0.45, fontFace: FONT, fontSize: 18,
    italic: true, color: C.muted, align: "center", margin: 0, isTextBox: true, objectName: "topic" });
  s.addText([
    { text: "M Naga Sai Dattu (23BCE0757)      Prakrit Bahl (23BCE0826)", options: { bold: true, breakLine: true } },
    { text: "School of Computer Science and Engineering (SCOPE)", options: { breakLine: true } },
    { text: "Vellore Institute of Technology, Vellore" },
  ], { x: X0, y: 4.6, w: CW, h: 1.2, fontFace: FONT, fontSize: 18, color: C.text, align: "center", margin: 0, isTextBox: true, objectName: "authors" });
  s.addNotes(`${PRAKRIT} Introduce both of us and the topic. Our project takes Paper 7, which we were given with its code, runs all of its methods on the real Kaggle data, and adds a new method that beats them. Prakrit covers the problem, the data and the methods (slides 1 to 8); Nagasai covers our method, the results and the code (slides 9 to 18).`);
}

// ---------------------------------------------------------------- 2 problem
content("Introduction", "The problem: patients who do not show up", [
  `In the real data, ${N.data.noshow} of booked patients did not come to their appointment`,
  "An empty slot is lost capacity, so clinics overbook: they give out more appointments than they have slots",
  { b: "Book too few extra patients: ", t: "slots stay idle and the waiting list grows" },
  { b: "Book too many: ", t: "when most patients come, the extra ones are seen in overtime" },
  "The right number depends on who is booked, so it changes from session to session",
  { b: "Question: ", t: "before each session, how many extra patients should the clinic book?" },
  "Each choice also changes the waiting list for later sessions, so this is a sequential decision problem, which reinforcement learning (RL) is built for [3]",
], `${PRAKRIT} Explain what the clinic has to balance. About one in five patients does not come. Booking extra patients fills those slots, but if everyone turns up the clinic works overtime. The decision repeats every session and affects the waiting list, which is why it is modelled as a Markov decision process and solved with RL.`);

// ---------------------------------------------------------------- 3 problem statement
{
  const s = titleOnly("Introduction", "Topic 7 problem statement and where we cover it",
    `${PRAKRIT} Walk through the problem statement line by line. Every requirement is met on the real data: the state uses the classifier's predicted risk band as asked, the action is the overbook level, the reward is Paper 7's reward, and all four evaluation items are reported in slides 10 to 14.`);
  table(s, [
    ["Requirement", "What we did"],
    ["Dataset: Medical Appointment No Shows", `All results use the real Kaggle table: ${N.data.appointments} appointments from 2016 [17]`],
    ["State: slots filled + predicted no-show risk band", `Slots filled (3 bands) × predicted risk band from a classifier (4) × waiting list band (5) = ${N.mdp.states} states`],
    ["Action: overbook level", `Book 0 to 4 extra patients before each session (${N.mdp.actions} actions)`],
    ["Reward: utilisation minus waiting and overtime", "shows − 3 × overtime − 0.5 × idle slots − 0.4 × waiting list + 0.3 × patients taken from the waiting list"],
    ["Futuristic angle: adapts to patient behaviour", "The policy reacts to each session's predicted risk and the waiting list; tested when the no-show rate rises or falls"],
    ["Evaluation: utilisation, wait time, overtime, MC vs Q-learning convergence", "All four reported on real scoring weeks (slides 10 to 14)"],
  ], X0, 1.4, [4.6, CW - 4.6], { size: 15, rowH: 0.62, leftCols: [1], name: "statement" });
}

// ---------------------------------------------------------------- 4 paper 7
{
  const s = titleOnly("Introduction", "Paper 7: what we were given",
    `${PRAKRIT} Paper 7 came with its code. It used a simulator written by hand because the Kaggle site could not be reached when it was written. Its learners were only ${N.paper7_gain.DP} to ${N.paper7_gain["Monte Carlo"]} above the rule while DP with ten times the data was ${N.paper7_gain["Oracle DP"]} above it, so the learners were held back by the data, not by the algorithm. Our job: run the same methods on the real data and beat them.`);
  heading(s, "What it did", X0, 1.4, 6.6);
  box(s, [
    "Modelled overbooking as an MDP with 100 states and 5 actions",
    "Ran on a simulator written by hand that copies the dataset's columns (the Kaggle site could not be reached when it was written)",
    `Learned from ${N.mdp.days} logged clinic days with DP, Monte Carlo (MC) control and Q-learning`,
    "Compared five exploration schedules and tested a change in the no-show rate",
  ], X0, 1.9, 6.6, 3.6, 17);
  heading(s, "What it found (simulator)", 7.6, 1.4, 5.1);
  table(s, [
    ["Method", "Return"],
    ["Guideline rule", N.paper7.Rule],
    ["DP on estimated MDP", N.paper7.DP],
    ["Monte Carlo control", N.paper7["Monte Carlo"]],
    ["Q-learning", N.paper7["Q-learning"]],
    ["Oracle DP (10x data)", N.paper7["Oracle DP"]],
  ], 7.6, 1.95, [3.4, 1.7], { size: 15, name: "paper7" });
  box(s, [
    { b: "The gap: ", t: `the learners were only ${N.paper7_gain.DP} to ${N.paper7_gain["Monte Carlo"]} above the rule, while DP with ten times the data was ${N.paper7_gain["Oracle DP"]} above it, and no real data was used` },
    { b: "Our task: ", t: "run Paper 7's methods on the real data, beat them with a new method, and report both in the same paper" },
  ], X0, 5.15, CW, 1.6, 17, "gap");
}

// ---------------------------------------------------------------- 5 data
pres.addSection({ title: "Data and method" });
content("Data and method", "The data: Medical Appointment No Shows (Kaggle)", [
  `${N.data.appointments} appointments of ${N.data.patients} patients in ${N.data.neighbourhoods} neighbourhoods in Brazil, on ${N.data.days} weekdays from 29 April to 8 June 2016 [17]`,
  `${N.data.noshow} of them were missed; the weekday hardly matters (${N.data.weekday_lo} to ${N.data.weekday_hi})`,
  `Booking lead time matters most: ${N.data.lead_same} no-shows when booked the same day, ${N.data.lead_31} when booked 31 or more days ahead`,
  { b: "No-show classifier: ", t: "gradient boosting [18], calibrated with isotonic regression [19], using scikit-learn [20]" },
  ["Inputs known at booking: lead time, age, sex, SMS reminder, welfare and condition flags, weekday, earlier appointments and earlier no-shows, neighbourhood", 1],
  ["Trained on appointments before 9 May, calibrated on 9 to 15 May, used from 16 May onwards (no later information leaks in)", 1],
  [`Test AUC ${N.data.auc}; mean predicted no-show ${N.data.pred} against ${N.data.obs} observed`, 1],
], `${PRAKRIT} Describe the table and what drives no-shows. The classifier gives each booked patient a no-show probability; the MDP state uses the average risk band of a session. It is trained only on earlier weeks, so nothing from the scoring weeks leaks in.`);

// ---------------------------------------------------------------- 6 MDP
content("Data and method", "From the table to an MDP", [
  { b: "Session: ", t: `all appointments of one neighbourhood on one day (at least 5), with ${N.mdp.slots} regular slots` },
  { b: "Episode: ", t: `${N.mdp.sessions} sessions drawn from real sessions, keeping the weekday order` },
  { b: "State: ", t: `slots filled (5 to 7, 8 to 9, 10) × predicted risk quartile (4) × waiting list band (5) = ${N.mdp.states} states` },
  { b: "Action: ", t: "book 0, 1, 2, 3 or 4 extra patients before the session" },
  { b: "Reward per session: ", t: "shows − 3 × overtime − 0.5 × idle − 0.4 × waiting list + 0.3 × taken from the waiting list (as in Paper 7)" },
  { b: "Attendance is real: ", t: "whether each booked patient comes is what the table records" },
  { b: "Time split: ", t: `learning sessions 16 to 25 May (${N.data.sessions_past}); scoring sessions 30 May to 8 June (${N.data.sessions_future}), never used for learning` },
  { b: "Learning data: ", t: `${N.mdp.days} logged clinic days, Paper 7's setting` },
], `${PRAKRIT} Explain how the table becomes an MDP. Every policy is learned on the May sessions and scored on the June sessions it never saw, which is how we can measure true performance on real data.`, 19);

// ---------------------------------------------------------------- 7 gaps
{
  const s = titleOnly("Data and method", "What the table cannot give, and what we did",
    `${PRAKRIT} Be open about this slide; it is the first thing an examiner will ask. Only attendance is real data. The decisions in the logs are generated with Paper 7's own behaviour policy, and the waiting list follows Paper 7's rule because the table has none. The paper says the same in Section 6.1 and in the limitations.`);
  table(s, [
    ["The table has no...", "So we..."],
    ["overbooking decisions", "generate the logs by running Paper 7's behaviour policy on the real sessions; who comes stays real"],
    ["waiting list", "keep Paper 7's waiting list rule (1.5 new patients per session); this is the one simulated part"],
    ["clinics or slots", `treat one neighbourhood on one day as a session with ${N.mdp.slots} regular slots`],
    ["times within a day", "count overtime and waiting in patients, as Paper 7 does"],
    ["more than 26 days", `draw ${N.mdp.days} logged days of ${N.mdp.sessions} sessions from the ${N.data.sessions_past} learning sessions`],
    ["way to change the no-show rate", `flip the recorded outcome of random patients: ${N.rates[0]} becomes ${N.rates[1]} or ${N.rates[2]}`],
  ], X0, 1.4, [3.6, CW - 3.6], { size: 16, rowH: 0.62, leftCols: [1], name: "gaps" });
}

// ---------------------------------------------------------------- 8 methods
{
  const s = titleOnly("Data and method", "Methods compared",
    `${PRAKRIT} Paper 7's three learners all estimate the full transition model from the logs. The baselines from published work are simplified versions, and their settings were tuned on the scoring weeks, which favours them. Every method is scored on the same ${N.mdp.episodes} episodes with the same random numbers. Hand over to Nagasai for our method.`);
  heading(s, "From Paper 7", X0, 1.4, 6.0);
  box(s, [
    "DP (value iteration) on the MDP estimated from the logs",
    "Monte Carlo control on that MDP",
    "Q-learning on that MDP",
    "Guideline rule (books 1 or 2 extra) and the clinician behaviour policy",
    `Oracle DP: DP fitted to ${N.mdp.oracle_days} days, ten times the data`,
  ], X0, 1.9, 6.0, 3.3, 17);
  heading(s, "Baselines from published work", 6.9, 1.4, 5.8);
  box(s, [
    "Single session optimiser: best k for the current session from each patient's risk [6]",
    "Threshold double booking (θ = 0.25): one extra booking per high risk patient [7]",
    "Fixed booking: always k = 2 extra",
  ], 6.9, 1.9, 5.8, 3.3, 17);
  box(s, [
    { b: "Scoring: ", t: `${N.mdp.seeds} replicates (independent logged datasets), ${N.mdp.episodes} episodes on the scoring sessions, the same random numbers for every policy; return = discounted reward over ${N.mdp.sessions} sessions` },
  ], X0, 4.75, CW, 1.2, 17, "scoring");
}

// ---------------------------------------------------------------- 9 our method
{
  const s = titleOnly("Data and method", "Our method: a structured model of the clinic",
    `${NAGASAI} Paper 7's learners estimate every transition separately, so the actions that matter (booking 3 or 4 extra) are hardly ever seen in the logs. But most of the clinic is known. We learn only patient behaviour, ${N.mdp.n_est} numbers that every logged session informs, and compute everything else from the clinic's rules. This is a factored MDP. The same MC and Q-learning can also run on it.`);
  heading(s, "Idea", X0, 1.4, 5.9);
  box(s, [
    `Paper 7's methods estimate every transition P(s' | s, a) from counts: ${N.mdp.n_emp} numbers from ${N.mdp.days} days`,
    "Booking 3 or 4 extra is rare in the logs, so those estimates are poor",
    "But most of the clinic is known: the slots, the reward, how extra bookings empty the waiting list",
    { b: "So we learn only patient behaviour: ", t: `${N.mdp.n_est} numbers` },
  ], X0, 1.9, 5.9, 4.6, 17);
  heading(s, "Steps", 6.85, 1.4, 5.9);
  box(s, [
    "Estimate from the logs: how often each slots and risk cell occurs (12), the waiting list arrival rate (1), and the waiting list lengths in each band (13)",
    "Compute each session's expected reward exactly from the patients' predicted show probabilities",
    "Assemble the transition model from these parts (a factored MDP [13])",
    "Solve it with DP (value iteration); MC and Q-learning can also learn on it",
  ], 6.85, 1.9, 5.9, 4.6, 17);
}

// ---------------------------------------------------------------- 10 main result
{
  pres.addSection({ title: "Results" });
  const s = titleOnly("Results", "Main result on the real scoring sessions",
    `${NAGASAI} Read the table from the top: the rule, the published baselines, Paper 7's learners, the oracle, then ours. Ours scores ${N.real["Ours: DP"]}: ${N.ours_vs.DP.d} over DP, ${N.ours_vs["Q-learning"].d} over Q-learning and ${N.ours_vs["Monte Carlo"].d} over MC (all ${N.ours_vs.DP.p}), and even ${N.ours_vs_ep["Oracle DP"].d} over the oracle with ten times the data. Against the single session optimiser the lead is small (${N.ours_vs_ep["Single session optimiser"].d}) but consistent; slide 13 shows when it grows. It also has the shortest waiting list.`);
  const rows = [["Method", "Return (± 95% CI)", "Utilisation", "Overtime per session", "Waiting list per session"]];
  const bold = [];
  N.table.forEach((r, i) => { rows.push([(r.paper7 ? "Paper 7: " : "") + r.name, r.J, r.util, r.over, r.wait]); if (r.ours && r.name.includes("DP")) bold.push(i + 1); });
  table(s, rows, X0, 1.3, [4.9, 2.1, 1.6, 1.75, 1.78], { size: 14, rowH: 0.34, bold, name: "results" });
  box(s, [
    { b: "Ours vs Paper 7: ", t: `${N.ours_vs.DP.d} over DP, ${N.ours_vs["Q-learning"].d} over Q-learning, ${N.ours_vs["Monte Carlo"].d} over MC (all ${N.ours_vs.DP.p}); ${N.ours_vs_ep["Oracle DP"].d} over the oracle with ten times the data` },
    { b: "Ours vs the best published baseline: ", t: `${N.ours_vs_ep["Single session optimiser"].d} (${N.ours_vs_ep["Single session optimiser"].p}), with the shortest waiting list of all methods` },
  ], X0, 6.0, CW, 0.95, 15, "takeaways");
}

// ---------------------------------------------------------------- 11 data size
{
  const s = titleOnly("Results", "How much logged data is needed",
    `${NAGASAI} This is the clearest picture of why our method wins. DP on the estimated MDP needs hundreds of days to approach its best, and with 25 days it is below the rule. The structured model reaches ${N.size[0].ours} with 25 days because it estimates ${N.mdp.n_est} numbers that every session informs.`);
  image(s, "fig_datasize", X0, 1.35, 7.4, 1011 / 644, "Return on the real scoring sessions against the number of logged clinic days, for DP on the estimated MDP and for the structured model");
  box(s, [
    `DP on the estimated MDP: ${N.size[0].dp.split(" ")[0]} with ${N.size[0].days} days (below the rule, ${N.real.Rule}), ${N.size[N.size.length - 1].dp.split(" ")[0]} with ${N.size[N.size.length - 1].days} days`,
    `Structured model: ${N.size[0].ours} at every size from ${N.size[0].days} to ${N.size[N.size.length - 1].days} days`,
    `Reason: it estimates ${N.mdp.n_est} numbers that every logged session informs, not ${N.mdp.n_emp} transition probabilities`,
    "Data coverage, not the learning algorithm, is what held Paper 7's methods back",
  ], 8.3, 1.5, 4.43, 5.2, 17);
}

// ---------------------------------------------------------------- 12 shift
{
  const s = titleOnly("Results", "Robustness to a change in the no-show rate",
    `${NAGASAI} The scoring sessions are rescored after flipping recorded outcomes, so the no-show rate rises to ${N.rates[1]} or falls to ${N.rates[2]}. The policies are not retrained. More no-shows hurt the rule most; ours stays ahead of every method at all three rates.`);
  const rows = [["Method", `Nominal (${N.rates[0]})`, `More no-shows (${N.rates[1]})`, `Fewer no-shows (${N.rates[2]})`]];
  const bold = [];
  N.shift.forEach((r, i) => { rows.push([r.name, ...r.vals]); if (r.ours) bold.push(i + 1); });
  table(s, rows, X0, 1.4, [4.1, 1.85, 2.2, 2.2], { size: 15, rowH: 0.42, bold, name: "shift" });
  box(s, [
    `Ours is ahead of every method at all three rates${N.shift_best_all ? "" : " (check)"}`,
    `More no-shows hurt the rule most: ${N.shift[0].vals[0]} falls to ${N.shift[0].vals[1]}`,
    "Policies are not retrained; only the patients' outcomes change",
  ], X0, 5.05, CW, 1.7, 17, "takeaways");
}

// ---------------------------------------------------------------- 13 demand
{
  const s = titleOnly("Results", "When planning across sessions matters: a busier clinic",
    `${NAGASAI} The single session optimiser only looks at the current session. When waiting costs more or more patients join the list, planning across sessions pays off: our lead grows from ${N.demand[0].gain} to ${N.demand[N.demand.length - 1].gain}. DP on the estimated MDP falls apart in the busiest setting because the actions it needs are rarely logged.`);
  const rows = [["Waiting cost", "New patients per session", "DP on estimated MDP", "Best fixed booking", "Single session optimiser", "Ours", "Our gain"]];
  N.demand.forEach((r) => rows.push([r.cost, r.arrivals, r.dp, r.fixed, r.single, r.ours, r.gain]));
  table(s, rows, X0, 1.4, [1.55, 1.95, 1.95, 1.75, 2.05, 1.3, 1.58], { size: 15, rowH: 0.5, name: "demand" });
  box(s, [
    "The single session optimiser looks only at the current session; ours also plans for the waiting list in later sessions",
    `The busier the clinic, the more that matters: our gain over it grows from ${N.demand[0].gain} to ${N.demand[N.demand.length - 1].gain}`,
    "Waiting cost 0.4 and 1.5 new patients per session is Paper 7's setting (slide 10)",
  ], X0, 4.35, CW, 2.3, 17, "takeaways");
}

// ---------------------------------------------------------------- 14 convergence
{
  const s = titleOnly("Results", "Convergence of Monte Carlo and Q-learning",
    `${NAGASAI} The problem statement asks for this comparison. The curves show the true return of the greedy policy as learning goes on. Q-learning ends higher than MC on both models, and both do better on our structured model. Paper 7's exploration schedule study, rerun on real data: no schedule gets 90% of the way to the optimum, so exploration is not the bottleneck; the data is.`);
  image(s, "fig_convergence", X0, 1.35, 6.6, 1178 / 762, "True return of the greedy policy during learning for Monte Carlo and Q-learning, on the estimated MDP and on the structured model");
  box(s, [
    `On the estimated MDP: MC ends at ${N.conv["Monte Carlo|sir"].final}, Q-learning at ${N.conv["Q-learning|sir"].final}`,
    `On our structured model: MC ${N.conv["Monte Carlo|ours"].final}, Q-learning ${N.conv["Q-learning|ours"].final}; its DP solution is ${N.real["Ours: DP"]}`,
    "Q-learning comes closer to the DP solution than MC on both",
    { b: "Exploration schedules: ", t: `no schedule reached 90% of the possible improvement in any of ${N.sched.runs} runs; near greedy was slowest for Q-learning (AUC ${N.sched.auc_ng} against ${N.sched.auc_best}); the ranking differs between MC and Q-learning (rank correlation ${N.sched.rho})` },
  ], 7.55, 1.4, W - X0 - 7.55, 5.4, 16);
}

// ---------------------------------------------------------------- 15 code
{
  pres.addSection({ title: "Code and conclusion" });
  const s = titleOnly("Code and conclusion", "Code: how it is built and run",
    `${NAGASAI} Everything is in the public repo, folder topic7. Paper 7's code is reused unchanged; our files only import and call it. One command reruns every experiment, and the paper is built from the results file, so no number in the paper or on these slides is typed by hand.`);
  heading(s, "Pipeline (Python)", X0, 1.4, 6.4);
  box(s, [
    { b: "noshow_data.py: ", t: "cleans the table, trains the classifier, builds sessions" },
    { b: "real_clinic.py: ", t: "the clinic MDP built from real sessions" },
    { b: "planner.py: ", t: "the structured model (our method)" },
    { b: "rl_core.py, common.py, ideas.py: ", t: "Paper 7's solvers and experiments, reused unchanged" },
    { b: "run_real.py: ", t: "runs every experiment, writes results.json and the figures" },
    { b: "paper07_v3.py, paper07_v4.py: ", t: "build the paper from results.json; check scripts verify it" },
  ], X0, 1.9, 6.4, 4.9, 16);
  heading(s, "Models and tools", 7.35, 1.4, 5.4);
  box(s, [
    "Gradient boosting classifier with isotonic calibration (scikit-learn)",
    "Value iteration, Monte Carlo control and Q-learning from Paper 7's rl_core",
    "numpy, scipy, matplotlib, python-docx",
    { b: "Repo: ", t: "github.com/Nagasai-Datta/ml_lab, folder topic7" },
    { b: "Reproduce: ", t: "python topic7/code/run_real.py (fixed seeds)" },
    { b: "Paper: ", t: "version 3 (full) and version 4 (without appendix questions and code)" },
  ], 7.35, 1.9, 5.4, 4.9, 16);
}

// ---------------------------------------------------------------- 16 limitations
content("Code and conclusion", "Limitations", [
  "Only attendance is real: the decisions in the logs are generated and the waiting list is simulated",
  "The structured model assumes the clinic knows its own rules (slots, reward weights, how the waiting list empties)",
  `Only about ${N.data.not_full} of real sessions have fewer than 10 bookings, so the slots filled part of the state rarely changes`,
  `The no-show rate drifted from ${N.data.noshow_past} to ${N.data.noshow_future}; the classifier predicts ${N.data.pred} against ${N.data.obs} observed`,
  "Waiting is measured by the length of the waiting list; the table has no times within a day",
  "The baselines from published work are simplified versions, tuned on the scoring sessions",
  "The results are about algorithms, not clinics; we make no clinical claim",
], `${NAGASAI} State these plainly. The first one matters most: the data records who came, not what a clinic decided, so the decisions are generated with Paper 7's behaviour policy. The method itself would need a clinic that can state its rules.`);

// ---------------------------------------------------------------- 17 conclusion
{
  const s = titleOnly("Code and conclusion", "Conclusion and future work",
    `${NAGASAI} Summarise in three sentences: Paper 7's methods work on real data but are limited by the logs; learning only patient behaviour removes that limit; and the advantage grows when the clinic is busier. Then the future work, then thank the audience and open questions.`);
  heading(s, "Conclusion", X0, 1.4, CW);
  box(s, [
    `On real sessions, Paper 7's DP, MC and Q-learning beat the rule (${N.real.DP}, ${N.real["Monte Carlo"]} and ${N.real["Q-learning"]} against ${N.real.Rule}) but stop short of the oracle (${N.real["Oracle DP"]}): the logged data is the limit`,
    `Learning only patient behaviour removes that limit: our structured model scores ${N.real["Ours: DP"]}, ahead of every method at all three no-show rates, with as few as ${N.size[0].days} logged days`,
    `Its lead over the best published baseline grows from ${N.demand[0].gain} to ${N.demand[N.demand.length - 1].gain} as the waiting list comes under pressure`,
  ], X0, 1.9, CW, 2.6, 17);
  heading(s, "Future work", X0, 4.0, CW);
  box(s, [
    "Logs in which the number of extra bookings is varied on purpose",
    "Overtime and waiting measured from visit times",
    "Reward weights and waiting list dynamics estimated from clinic records instead of assumed",
  ], X0, 4.5, CW, 1.8, 17, "future");
}

// ---------------------------------------------------------------- 18 references
{
  const s = titleOnly("Code and conclusion", "References",
    `${NAGASAI} References are numbered as in the paper. The dataset is [17]; the published baselines are [6] and [7]; the factored MDP idea behind our method is [13].`);
  const half = Math.ceil(N.refs.length / 2);
  const col = (list) => list.map((r, i) => ({ text: r, options: { fontSize: 12, fontFace: FONT, color: C.text, paraSpaceAfter: 4, breakLine: i < list.length - 1 } }));
  s.addText(col(N.refs.slice(0, half)), { x: X0, y: 1.3, w: 5.95, h: 5.55, valign: "top", margin: 0, isTextBox: true, objectName: "refs_left" });
  s.addText(col(N.refs.slice(half)), { x: X0 + 6.18, y: 1.3, w: 5.95, h: 5.55, valign: "top", margin: 0, isTextBox: true, objectName: "refs_right" });
}

// ---------------------------------------------------------------- backup: expected questions
{
  pres.addSection({ title: "Backup" });
  const pick = [3, 7, 8, 9, 10, 1];                       // question numbers from the paper's Appendix A (version 3)
  for (let i = 0; i < pick.length; i += 2) {
    const s = titleOnly("Backup", `Backup: questions we expect (${i / 2 + 1} of ${pick.length / 2})`,
      "Backup slide for questions. The answers are the same as in Appendix A of version 3 of the paper.");
    [pick[i], pick[i + 1]].forEach((q, j) => {
      const qa = N.qa[q - 1], y = 1.4 + j * 2.75;
      // point to the slides rather than to the paper's own section and table numbers
      const ans = qa.a.replace("which Section 9 lists as a limitation", "which is one of our limitations (slide 16)")
        .replace("(Table 9)", "(slide 13)").replace("(Section 6.1)", "(slide 7)");
      s.addText(qa.q, { x: X0, y, w: CW, h: 0.45, fontFace: FONT, fontSize: 18, bold: true, color: C.title, margin: 0, isTextBox: true, objectName: `q${q}` });
      s.addText(ans, { x: X0, y: y + 0.5, w: CW, h: 2.05, fontFace: FONT, fontSize: 16, color: C.text, valign: "top", margin: 0, isTextBox: true, objectName: `a${q}` });
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
