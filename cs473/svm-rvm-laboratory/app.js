"use strict";

const byId = id => document.getElementById(id);
const steps = [
  ["Shared candidates", "Both models start with one possible RBF center per training example. The sensor data are fabricated. Blue circles are normal readings and orange squares are faults."],
  ["Fit the SVM", "The SVM balances a wide margin against violations. A point with nonzero dual coefficient contributes to the score. Its signed weight is αᵢyᵢ. Try changing C to see which points qualify."],
  ["Fit the RVM", "The RVM learns a separate prior precision for each weight. Higher precision concentrates that weight near zero. Drag the round slider back to the start and watch centers disappear as their precisions cross the pruning threshold."],
  ["Trace a prediction", "Move the crosshair. Both models use the same kernel value for any given center, but their weights differ. Expand your view down to the table to see what pushes the scores in each direction."],
  ["Compare prediction work", "Each model evaluates one kernel per retained center. Compare the counts at the final fitting round, then change the sensor pattern or γ. The RVM and SVM can retain different points even when the predictions agree."]
];

let stage = 0;
let points, parameters, svm, history, support;
const grid = { left: 38, top: 10, size: 270 };
const number = value => value.toFixed(3);
const coordinate = id => Number(byId(id).value);
const valueAt = (model, kernelValues) => model.centers.reduce((sum, i, j) =>
  sum + model.weights[j] * kernelValues[i], model.bias
);
const kernelsAt = x => points.map(point => SVMLab.kernel(point.x, x, parameters));
const px = x => grid.left + (x + 2) * grid.size / 4;
const py = y => grid.top + (2 - y) * grid.size / 4;

function train() {
  parameters = { type: "rbf", gamma: coordinate("gamma"), C: coordinate("cost") };
  points = SVMLab.data(byId("dataset").value, 32).map(point => ({ ...point, y: -point.y }));
  svm = SVMLab.fit(points, parameters);
  const centers = points.map((_, i) => i).filter(i => svm.a[i] > 1e-8);
  support = { centers, weights: centers.map(i => svm.a[i] * points[i].y), bias: svm.b };
  history = RVMLab.fit(svm.K, points.map(point => point.y));
  byId("round").value = history.length - 1;
  render();
}

function plot(model, probe) {
  const colors = ["var(--blue)", "var(--orange)"];
  const parts = ['<rect x="38" y="10" width="270" height="270" rx="4" fill="var(--soft)"/>'];
  if (model) {
    const cells = 36;
    const size = grid.size / cells;
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        const v = valueAt(model, kernelsAt([4 * (x + .5) / cells - 2, 2 - 4 * (y + .5) / cells]));
        parts.push('<rect x="' + (grid.left + x * size) + '" y="' + (grid.top + y * size) +
          '" width="' + size + '" height="' + size + '" fill="' + colors[Number(v >= 0)] +
          '" opacity="' + (.10 + .11 * Math.min(1, Math.abs(v))) + '"/>');
      }
    }
  }
  for (const tick of [-1, 0, 1]) {
    parts.push('<path d="M' + px(tick) + ' 10V280 M38 ' + py(tick) + 'H308" stroke="var(--line)" opacity=".7"/>');
  }
  for (const point of points) {
    const x = px(point.x[0]), y = py(point.x[1]);
    const active = model && model.centers.includes(point.id);
    if (active) parts.push('<circle cx="' + x + '" cy="' + y + '" r="8" fill="none" stroke="var(--text)" stroke-width="1.6"/>');
    const style = ' fill="' + colors[Number(point.y === 1)] + '" opacity="' + (model && !active ? .42 : 1) + '">';
    const title = '<title>Center ' + (point.id + 1) + (active ? ' · retained' : '') + '</title>';
    parts.push(point.y === 1
      ? '<rect x="' + (x - 4) + '" y="' + (y - 4) + '" width="8" height="8"' + style + title + '</rect>'
      : '<circle cx="' + x + '" cy="' + y + '" r="4.2"' + style + title + '</circle>');
  }
  const x = px(probe[0]), y = py(probe[1]);
  parts.push('<path d="M' + (x - 9) + ' ' + y + 'h18 M' + x + ' ' + (y - 9) + 'v18" stroke="var(--text)" stroke-width="2"/>');
  parts.push('<g fill="var(--muted)" font-size="11" text-anchor="middle"><text x="173" y="306">Vibration (standardized)</text><text transform="translate(15 145) rotate(-90)">Temperature (standardized)</text></g>');
  return parts.join("");
}

function render() {
  for (const id of ["gamma", "cost", "round", "probe-x", "probe-y"]) byId(id + "-value").textContent = byId(id).value;
  byId("steps").innerHTML = steps.map(([title], i) =>
    '<li' + (stage === i ? ' class="active" aria-current="step"' : '') + '>' + (i + 1) + '. ' + title + '</li>'
  ).join("");
  byId("explanation").innerHTML = '<div class="eyebrow">STEP ' + (stage + 1) + ' OF 5</div><h2>' +
    steps[stage][0] + '</h2><p>' + steps[stage][1] + '</p>';
  byId("back").disabled = stage === 0;
  byId("next").textContent = stage === steps.length - 1 ? "Restart walkthrough" : "Next step";
  byId("round-control").hidden = stage !== 2;
  byId("trace").hidden = stage < 3;

  const probe = [coordinate("probe-x"), coordinate("probe-y")];
  const kernelValues = kernelsAt(probe);
  const relevance = history[stage === 2 ? coordinate("round") : history.length - 1];
  const models = [
    ["svm", stage >= 1 ? support : null, "Support vectors"],
    ["rvm", stage >= 2 ? relevance : null, "Active centers"]
  ];
  for (const [id, model, label] of models) {
    byId(id + "-plot").innerHTML = plot(model, probe);
    byId(id + "-count").textContent = model
      ? label + " · " + model.centers.length + " / " + points.length
      : points.length + " candidate centers";
    const score = model ? valueAt(model, kernelValues) : null;
    byId(id + "-score").textContent = model
      ? "Probe score " + number(score) + " · " + (score >= 0 ? "Fault" : "Normal")
      : "Decision regions appear after fitting.";
  }

  if (stage === 2) {
    const strongest = Math.max(...relevance.alpha);
    byId("explanation").insertAdjacentHTML("beforeend", "<p>Round " + coordinate("round") + ": " +
      (points.length - relevance.centers.length) + " pruned. Largest remaining center precision: " +
      (Number.isFinite(strongest) ? number(strongest) : "no centers remain") + ".</p>");
  }
  if (stage < 3) return;

  byId("trace-caption").textContent = "Kernel evaluations per prediction · SVM " + support.centers.length +
    " · RVM " + relevance.centers.length + ". A dash marks a center the model dropped.";
  const active = [...new Set([...support.centers, ...relevance.centers])].sort((a, b) => a - b);
  const cells = values => "<tr>" + values.map((v, j) =>
    j === 0 ? '<th scope="row">' + v + "</th>" : "<td>" + v + "</td>"
  ).join("") + "</tr>";
  const terms = (model, i) => {
    const index = model.centers.indexOf(i);
    if (index < 0) return ["—", "—"];
    const weight = model.weights[index];
    return [number(weight), number(weight * kernelValues[i])];
  };
  byId("terms").innerHTML = cells(["Intercept", "—", "—", number(support.bias), "—", number(relevance.bias)]) +
    active.map(i => cells(["#" + (i + 1), number(kernelValues[i]), ...terms(support, i), ...terms(relevance, i)])).join("") +
    cells(["Total score", "—", "—", number(valueAt(support, kernelValues)), "—", number(valueAt(relevance, kernelValues))]);
}

byId("next").addEventListener("click", () => { stage = (stage + 1) % steps.length; render(); });
byId("back").addEventListener("click", () => { stage--; render(); });
for (const id of ["dataset", "gamma", "cost"]) byId(id).addEventListener("change", train);
for (const id of ["gamma", "cost"]) byId(id).addEventListener("input", () => { byId(id + "-value").textContent = byId(id).value; });
for (const id of ["probe-x", "probe-y", "round"]) byId(id).addEventListener("input", render);
for (const id of ["svm-plot", "rvm-plot"]) {
  const svg = byId(id);
  svg.addEventListener("click", event => {
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const local = point.matrixTransform(svg.getScreenCTM().inverse());
    if (local.x < grid.left || local.x > grid.left + grid.size ||
        local.y < grid.top || local.y > grid.top + grid.size) return;
    byId("probe-x").value = (4 * (local.x - grid.left) / grid.size - 2).toFixed(2);
    byId("probe-y").value = (2 - 4 * (local.y - grid.top) / grid.size).toFixed(2);
    render();
  });
}

const questions = [
  ["What is shared when both models score a test point?", ["The coefficient at each center", "The RBF kernel value at each center", "The list of retained centers"], 1, "The kernel depends on the two inputs and γ. Each model learns its own weights."],
  ["Which training points contribute to the SVM score?", ["Points with nonzero dual coefficients", "Only misclassified points", "Only points with the highest prior precision"], 0, "A correctly classified point on the margin can also be a support vector."],
  ["What happens as an RVM weight’s prior precision grows?", ["Its prior variance grows", "The RBF kernel becomes wider", "Its prior concentrates around zero"], 2, "Variance equals 1 divided by precision. A sufficiently large precision triggers pruning in this lab."],
  ["One model retains 5 centers out of 32. How many kernel evaluations does it need for one new score?", ["32", "5", "27"], 1, "Only the five retained terms enter the score. The intercept is added separately."],
  ["What changes when you increase C while keeping the data and γ fixed?", ["The SVM pays more for margin violations", "The RVM prunes at a lower precision", "Both models use a narrower RBF kernel"], 0, "C is an SVM training parameter. The shared RBF kernel still uses the same γ."]
];

questions.forEach(([question, choices, correct, explanation], i) => {
  const section = document.createElement("section");
  section.className = "question";
  section.innerHTML = "<h3>" + (i + 1) + ". " + question + "</h3>" +
    choices.map(choice => "<button>" + choice + "</button>").join("") +
    '<p role="status"></p>';
  const buttons = [...section.querySelectorAll("button")];
  buttons.forEach((button, index) => button.addEventListener("click", () => {
    buttons.forEach(option => option.classList.remove("correct", "wrong"));
    button.classList.add(index === correct ? "correct" : "wrong");
    section.querySelector("p").textContent = (index === correct ? "Correct. " : "Try again. ") + explanation;
  }));
  byId("quiz").append(section);
});

train();
