(function (root) {
  "use strict";
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);

  // Invert a positive definite precision matrix using its Cholesky factor.
  function inverse(matrix) {
    const n = matrix.length;
    const lower = matrix.map(row => row.map(() => 0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let value = matrix[i][j];
        for (let k = 0; k < j; k++) value -= lower[i][k] * lower[j][k];
        lower[i][j] = i === j ? Math.sqrt(value) : value / lower[j][j];
      }
    }
    const result = matrix.map(row => row.map(() => 0));
    for (let j = 0; j < n; j++) {
      const forward = Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        let value = Number(i === j);
        for (let k = 0; k < i; k++) value -= lower[i][k] * forward[k];
        forward[i] = value / lower[i][i];
      }
      for (let i = n - 1; i >= 0; i--) {
        let value = forward[i];
        for (let k = i + 1; k < n; k++) value -= lower[k][i] * result[k][j];
        result[i][j] = value / lower[i][i];
      }
    }
    return result;
  }

  function fit(kernelMatrix, labels) {
    // Index -1 denotes the intercept, with a fixed weak Gaussian prior.
    let centers = [-1, ...labels.map((_, i) => i)];
    let alpha = centers.map(i => i === -1 ? .01 : 1);
    let weights = centers.map(() => 0);
    const history = [];
    const cutoff = 1e4;

    function posterior() {
      const design = kernelMatrix.map(row => centers.map(i => i === -1 ? 1 : row[i]));
      function objective(candidate) {
        return design.reduce((total, row, i) => {
          const z = -labels[i] * dot(row, candidate);
          return total + Math.max(0, z) + Math.log1p(Math.exp(-Math.abs(z)));
        }, 0) + .5 * dot(alpha, candidate.map(w => w * w));
      }
      let covariance;
      for (let iteration = 0; iteration < 35; iteration++) {
        const probabilities = design.map(row => sigmoid(dot(row, weights)));
        const curvature = probabilities.map(p => p * (1 - p));
        const precision = alpha.map((a, j) => alpha.map((_, k) =>
          design.reduce((sum, row, i) => sum + row[j] * curvature[i] * row[k], j === k ? a : 0)
        ));
        covariance = inverse(precision);
        const gradient = alpha.map((a, j) => design.reduce((sum, row, i) =>
          sum + row[j] * (Number(labels[i] === 1) - probabilities[i]), -a * weights[j]
        ));
        const direction = covariance.map(row => dot(row, gradient));
        if (Math.max(...direction.map(Math.abs)) < 1e-7 || iteration === 34) break;
        const current = objective(weights);
        for (let step = 1; step > 1e-7; step /= 2) {
          const candidate = weights.map((w, j) => w + step * direction[j]);
          if (objective(candidate) <= current) {
            weights = candidate;
            break;
          }
        }
      }
      return covariance;
    }

    function record() {
      history.push({
        centers: centers.slice(1),
        alpha: alpha.slice(1),
        weights: weights.slice(1),
        bias: weights[0]
      });
    }

    for (let round = 0; round < 40; round++) {
      const covariance = posterior();
      record();
      // MacKay fixed-point update with a Laplace posterior for classification.
      const updated = alpha.map((a, j) => j === 0 ? a :
        Math.max(1e-4, (1 - a * covariance[j][j]) / Math.max(1e-20, weights[j] ** 2))
      );
      const keep = centers.map((_, j) => j).filter(j => updated[j] < cutoff);
      centers = keep.map(j => centers[j]);
      weights = keep.map(j => weights[j]);
      alpha = keep.map(j => updated[j]);
    }
    posterior();
    record();
    return history;
  }
  if (typeof module !== "undefined") module.exports = { fit };
  else root.RVMLab = { fit };
})(typeof window !== "undefined" ? window : globalThis);
