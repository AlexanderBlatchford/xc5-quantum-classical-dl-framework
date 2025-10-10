export type Point = { x: number; y: number }

function binarize(yTrue: number[], nClasses: number) {
  const m = yTrue.length
  const Y = Array.from({ length: m }, () => new Array<number>(nClasses).fill(0))
  for (let i = 0; i < m; i++) Y[i][yTrue[i]] = 1
  return Y
}

export function microRoc(yTrue: number[], proba: number[][], steps = 101): { curve: Point[] } {
  const n = yTrue.length
  if (!n || !proba.length) return { curve: [] }
  const nClasses = proba[0].length
  const Y = binarize(yTrue, nClasses)

  const curve: Point[] = []
  for (let si = 0; si < steps; si++) {
    const thr = si / (steps - 1)
    let TP = 0, FP = 0, TN = 0, FN = 0
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < nClasses; c++) {
        const score = proba[i][c]
        const y = Y[i][c]
        const pred = score >= thr ? 1 : 0
        if (pred === 1 && y === 1) TP++
        else if (pred === 1 && y === 0) FP++
        else if (pred === 0 && y === 0) TN++
        else FN++
      }
    }
    const TPR = TP + FN ? TP / (TP + FN) : 0
    const FPR = FP + TN ? FP / (FP + TN) : 0
    curve.push({ x: FPR, y: TPR })
  }
  return { curve }
}

export function microPr(yTrue: number[], proba: number[][], steps = 101): { curve: Point[] } {
  const n = yTrue.length
  if (!n || !proba.length) return { curve: [] }
  const nClasses = proba[0].length
  const Y = binarize(yTrue, nClasses)

  const curve: Point[] = []
  for (let si = 0; si < steps; si++) {
    const thr = si / (steps - 1)
    let TP = 0, FP = 0, FN = 0
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < nClasses; c++) {
        const score = proba[i][c]
        const y = Y[i][c]
        const pred = score >= thr ? 1 : 0
        if (pred === 1 && y === 1) TP++
        else if (pred === 1 && y === 0) FP++
        else if (pred === 0 && y === 1) FN++
      }
    }
    const precision = TP + FP ? TP / (TP + FP) : 1
    const recall = TP + FN ? TP / (TP + FN) : 0
    curve.push({ x: recall, y: precision })
  }
  return { curve }
}
