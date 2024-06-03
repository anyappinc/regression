package regression

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/anyappinc/regression/v2/logger"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

var (
	// ErrNotEnoughObservations signals that there weren't enough observations to train the model.
	ErrNotEnoughObservations = errors.New("not enough observations")
	// ErrTooManyExplanatoryVars signals that there are too many explanatory variables for the number of observations being made.
	ErrTooManyExplanatoryVars = errors.New("not enough observations to support this many explanatory variables")
	// ErrNoExplanatoryVars signals that there is no explanatory variables to train the model.
	ErrNoExplanatoryVars = errors.New("no explanatory variables to train the models")
	// ErrInvalidArgument signals that any of given arguments to call the function was invalid.
	ErrInvalidArgument = errors.New("invalid argument")
)

type observation struct {
	objectiveVar    float64   // 目的変数
	explanatoryVars []float64 // 説明変数
}

// NewObservation creates a well formed *observation used for training.
func NewObservation(o float64, es []float64) *observation {
	return &observation{objectiveVar: o, explanatoryVars: es}
}

func calcPredictedVal(observations []float64, coeffs []float64, intercept float64) (float64, error) {
	if len(observations) != len(coeffs) {
		return 0, ErrInvalidArgument
	}
	var p float64
	for i, obs := range observations {
		p += obs * coeffs[i]
	}
	return p + intercept, nil
}

// Regression is the exposed data structure for interacting with the API.
type Regression struct {
	objectiveVars                  []float64        // 目的変数の観測値
	objectiveVarLabel              *string          // 目的変数の名称
	explanatoryVarsMatrix          [][]float64      // 説明変数、各行が1説明変数、各列に説明変数ごとの観測値
	explanatoryVarsLabelMap        map[int]string   // 説明変数の名称
	disregardingExplanatoryVarsSet map[int]struct{} // 分析に使わない説明変数のインデックスのセット
}

// NewRegression initializes the structure and returns it for interacting with regression APIs.
func NewRegression() *Regression {
	return &Regression{
		explanatoryVarsLabelMap:        map[int]string{},
		disregardingExplanatoryVarsSet: map[int]struct{}{},
	}
}

// SetObjectiveVariableLabel sets the label of the objective variable.
func (r *Regression) SetObjectiveVariableLabel(label string) {
	r.objectiveVarLabel = &label
}

// GetObjectiveVariableLabel gets the label of the objective variable.
func (r *Regression) GetObjectiveVariableLabel() string {
	if r.objectiveVarLabel == nil {
		return "Y"
	}
	return *r.objectiveVarLabel
}

// SetExplanatoryVariableLabel sets the label of i-th explanatory variable.
func (r *Regression) SetExplanatoryVariableLabel(i int, label string) {
	r.explanatoryVarsLabelMap[i] = label
}

// GetExplanatoryVariableLabel gets the label of i-th explanatory variable.
func (r *Regression) GetExplanatoryVariableLabel(i int) string {
	label, ok := r.explanatoryVarsLabelMap[i]
	if !ok {
		return "X" + strconv.Itoa(i)
	}
	return label
}

// DisregardIndex adds given index to the disregarding set
func (r *Regression) DisregardIndex(idx int) {
	r.disregardingExplanatoryVarsSet[idx] = struct{}{}
}

// ResetDisregarding : 無視する説明変数の設定をリセットする
func (r *Regression) ResetDisregarding() {
	r.disregardingExplanatoryVarsSet = map[int]struct{}{}
}

// AddObservations adds observations.
func (r *Regression) AddObservations(observations ...*observation) error {
	if observations == nil {
		return nil
	}
	numOfExplanatoryVars := len(observations[0].explanatoryVars)
	if numOfExplanatoryVars == 0 {
		return ErrInvalidArgument
	}
	// すべての観測値の説明変数の数が一致していることを確認
	for _, obs := range observations[1:] {
		if len(obs.explanatoryVars) != numOfExplanatoryVars {
			return ErrInvalidArgument
		}
	}
	if r.explanatoryVarsMatrix != nil {
		// 観測値の説明変数の数と既にセットされている説明変数の数が一致していることを確認
		if numOfExplanatoryVars != len(r.explanatoryVarsMatrix) {
			return ErrInvalidArgument
		}
	} else {
		// 初期化
		r.explanatoryVarsMatrix = make([][]float64, numOfExplanatoryVars)
	}

	for _, obs := range observations {
		r.objectiveVars = append(r.objectiveVars, obs.objectiveVar)
		for i, ev := range obs.explanatoryVars {
			r.explanatoryVarsMatrix[i] = append(r.explanatoryVarsMatrix[i], ev)
		}
	}
	return nil
}

func (r *Regression) run() (*basicRawModel, error) {
	numOfObservations := len(r.objectiveVars) // == len(r.explanatoryVarsMatrix[n])
	if numOfObservations <= 2 {
		return nil, ErrNotEnoughObservations
	}

	disregardedExplanatoryVarsSet := map[int]struct{}{}
	numOfExplanatoryVars := func() int {
		_numOfExplanatoryVars, _numToIgnore := len(r.explanatoryVarsMatrix), len(r.disregardingExplanatoryVarsSet)
		for idx := range r.disregardingExplanatoryVarsSet {
			if idx < 0 || idx >= _numOfExplanatoryVars {
				logger.Warn.Printf("Cannot ignore index %d: index out of range [0:%d]", idx, _numOfExplanatoryVars)
				_numToIgnore--
			}
			disregardedExplanatoryVarsSet[idx] = struct{}{}
		}
		return _numOfExplanatoryVars - _numToIgnore
	}()

	if numOfExplanatoryVars == 0 {
		return nil, ErrNoExplanatoryVars
	} else if numOfExplanatoryVars >= numOfObservations {
		return nil, ErrTooManyExplanatoryVars
	}

	// 元のインデックスと実際のインデックスの対応表を作成する
	// 説明変数の名称を抽出する
	indexesTable, explanatoryVarsLabels := make([]int, 0, numOfExplanatoryVars), make([]string, 0, numOfExplanatoryVars)
	for idx := range r.explanatoryVarsMatrix {
		if _, ok := r.disregardingExplanatoryVarsSet[idx]; ok {
			continue
		}
		indexesTable = append(indexesTable, idx)
		explanatoryVarsLabels = append(explanatoryVarsLabels, r.GetExplanatoryVariableLabel(idx))
	}

	// 目的変数の観測値の平均と標準偏差
	meanOfObjectiveVars, standardDeviationOfObjectiveVars := stat.MeanStdDev(r.objectiveVars, nil)

	// 各説明変数の観測値の平均と標準偏差
	meansOfExplanatoryVars, standardDeviationOfExplanatoryVars := make([]float64, 0, numOfExplanatoryVars), make([]float64, 0, numOfExplanatoryVars)
	for idx, explanatoryVars := range r.explanatoryVarsMatrix {
		if _, ok := r.disregardingExplanatoryVarsSet[idx]; ok {
			continue
		}
		mean, standardDeviation := stat.MeanStdDev(explanatoryVars, nil)
		meansOfExplanatoryVars = append(meansOfExplanatoryVars, mean)
		standardDeviationOfExplanatoryVars = append(standardDeviationOfExplanatoryVars, standardDeviation)
	}

	objectiveVarsDense := mat.NewDense(numOfObservations, 1, r.objectiveVars)
	colLen := numOfExplanatoryVars + 1 // +1: 定数項
	explanatoryVarsDense := func() *mat.Dense {
		d := mat.NewDense(numOfObservations, colLen, nil)
		var row int
		for idx, ev := range r.explanatoryVarsMatrix {
			if _, ok := r.disregardingExplanatoryVarsSet[idx]; ok {
				continue
			}
			for col, ov := range ev {
				d.Set(col, row, ov) // 転置する
			}
			row++
		}
		// 定数項を1で初期化する
		for i := 0; i < numOfObservations; i++ {
			d.Set(i, colLen-1, 1)
		}
		return d
	}()

	// 偏回帰係数とy切片を算出する
	qr, qrQ, qrR, qTY := new(mat.QR), new(mat.Dense), new(mat.Dense), new(mat.Dense)
	qr.Factorize(explanatoryVarsDense)
	qr.QTo(qrQ) // 直交行列 Q
	qr.RTo(qrR) // 上三角行列 R
	qTY.Mul(qrQ.T(), objectiveVarsDense)
	// ここで`qrR`は上三角行列なので
	// GaussJordanの掃き出し法の後退代入で各係数を算出することができる
	// （逆行列を計算する必要がない）
	_coeffs := make([]float64, colLen)
	for i := len(_coeffs) - 1; i >= 0; i-- {
		_coeffs[i] = qTY.At(i, 0)
		for j := i + 1; j < len(_coeffs); j++ {
			_coeffs[i] -= _coeffs[j] * qrR.At(i, j)
		}
		_coeffs[i] /= qrR.At(i, i)
	}
	intercept := _coeffs[len(_coeffs)-1] // y切片
	coeffs := _coeffs[:len(_coeffs)-1]   // 係数

	// 予測値
	predictedVals := make([]float64, numOfObservations)
	for i := range predictedVals {
		val, err := calcPredictedVal(explanatoryVarsDense.RawRowView(i)[:numOfExplanatoryVars], coeffs, intercept)
		if err != nil {
			return nil, err
		}
		predictedVals[i] = val
	}

	// 残差
	residuals := make([]float64, numOfObservations)
	for i, ov := range r.objectiveVars {
		residuals[i] = ov - predictedVals[i]
	}

	// 残差変動
	var unexplainedVariation float64
	for _, residual := range residuals {
		unexplainedVariation += math.Pow(residual, 2)
	}

	// 回帰変動
	var explainedVariation float64
	for _, pv := range predictedVals {
		explainedVariation += math.Pow(pv-meanOfObjectiveVars, 2)
	}

	// 全変動
	totalVariation := unexplainedVariation + explainedVariation

	// 決定係数
	r2 := 1 - unexplainedVariation/totalVariation

	return &basicRawModel{
		indexesTable:                       indexesTable,
		objectiveVarLabel:                  r.GetObjectiveVariableLabel(),
		explanatoryVarsLabels:              explanatoryVarsLabels,
		objectiveVarsDense:                 objectiveVarsDense,
		explanatoryVarsDense:               explanatoryVarsDense,
		numOfObservations:                  numOfObservations,
		numOfExplanatoryVars:               numOfExplanatoryVars,
		disregardedExplanatoryVarsSet:      disregardedExplanatoryVarsSet,
		meanOfObjectiveVars:                meanOfObjectiveVars,
		standardDeviationOfObjectiveVars:   standardDeviationOfObjectiveVars,
		meansOfExplanatoryVars:             meansOfExplanatoryVars,
		standardDeviationOfExplanatoryVars: standardDeviationOfExplanatoryVars,
		coeffs:                             coeffs,
		intercept:                          intercept,
		predictedVals:                      predictedVals,
		residuals:                          residuals,
		totalVariation:                     totalVariation,
		explainedVariation:                 explainedVariation,
		unexplainedVariation:               unexplainedVariation,
		r:                                  math.Sqrt(r2),
		r2:                                 r2,
	}, nil
}

// Run calculates a model using QR decomposition.
func (r *Regression) Run() (*Model, error) {
	bm, err := r.run()
	if err != nil {
		return nil, err
	}

	// 自由度
	// `bm.numOfObservations`が2より大きいことと
	// `bm.numOfExplanatoryVars`が`bm.numOfObservations`より小さいことはすでに保証されている
	totalDegreeOfFreedom := float64(bm.numOfObservations - 1)
	regressionDegreeOfFreedom := float64(bm.numOfExplanatoryVars)
	residualDegreeOfFreedom := totalDegreeOfFreedom - regressionDegreeOfFreedom

	// 説明変数の数が観測値の数のよりちょうど1少ないとき、
	// `residualDegreeOfFreedom`（残差自由度）が0になる
	// ただし条件式は`gonum`内部の`cephes.Incbet`に合わせる
	if residualDegreeOfFreedom <= 0 {
		return nil, ErrTooManyExplanatoryVars
	}

	// 自由度調整済み決定係数
	adjustedR2 := 1 - (bm.unexplainedVariation/residualDegreeOfFreedom)/(bm.totalVariation/totalDegreeOfFreedom)

	// 回帰の平方和 = 回帰変動
	regressionSumOfSquares := bm.explainedVariation

	// F検定
	regressionFstat := (bm.explainedVariation / regressionDegreeOfFreedom) / (bm.unexplainedVariation / residualDegreeOfFreedom)

	// 回帰の有意確率: Prob (F-statistic)
	regressionProb := distuv.F{
		D1: regressionDegreeOfFreedom,
		D2: residualDegreeOfFreedom,
	}.Survival(regressionFstat)

	// 残差の平方和
	var residualSumOfSquares float64
	for _, residual := range bm.residuals {
		residualSumOfSquares += math.Pow(residual, 2)
	}

	// 残差の分散（誤差の分散の不偏推定）
	residualsVariance := residualSumOfSquares / residualDegreeOfFreedom

	// 回帰の標準誤差（推定値の標準偏差）
	standardError := math.Sqrt(residualsVariance)

	// 許容度 及び 分散拡大係数（VIF）
	coeffsTolerances, coeffsVIFs := make([]float64, 0, bm.numOfExplanatoryVars), make([]float64, 0, bm.numOfExplanatoryVars)
	if bm.numOfExplanatoryVars < 2 {
		coeffsTolerances, coeffsVIFs = make([]float64, bm.numOfExplanatoryVars), make([]float64, bm.numOfExplanatoryVars)
	} else {
		for idx, explanatoryVars := range r.explanatoryVarsMatrix {
			if _, ok := r.disregardingExplanatoryVarsSet[idx]; ok {
				continue
			}
			r.disregardingExplanatoryVarsSet[idx] = struct{}{}
			bm, err := (&Regression{
				objectiveVars:                  explanatoryVars,
				explanatoryVarsMatrix:          r.explanatoryVarsMatrix,
				disregardingExplanatoryVarsSet: r.disregardingExplanatoryVarsSet,
			}).run()
			if err != nil {
				panic(err) // Error should never happens
			}
			delete(r.disregardingExplanatoryVarsSet, idx)

			tolerance := 1 - bm.r2
			coeffsTolerances = append(coeffsTolerances, tolerance)
			coeffsVIFs = append(coeffsVIFs, 1/tolerance)
		}
	}

	// 説明変数の観測値の残差の行列
	explanatoryVarsResidualsDense := mat.NewDense(bm.numOfObservations, bm.numOfExplanatoryVars, nil)
	explanatoryVarsResidualsDense.Apply(func(i, j int, v float64) float64 {
		return v - bm.meansOfExplanatoryVars[j]
	}, bm.explanatoryVarsDense.Slice(0, bm.numOfObservations, 0, bm.numOfExplanatoryVars))

	// 説明変数の偏差平方和積和行列
	s := new(mat.Dense)
	s.Mul(explanatoryVarsResidualsDense.T(), explanatoryVarsResidualsDense)

	sQR, sQ, sR := new(mat.QR), new(mat.Dense), new(mat.Dense)
	sQR.Factorize(s)
	sQR.QTo(sQ) // 直交行列 Q
	sQR.RTo(sR) // 上三角行列 R

	sRInv := new(mat.Dense)
	if err := sRInv.Inverse(sR); err != nil {
		e := fmt.Errorf("cannot inverse a matrix(sR): %w", err)

		logger.Err.Println(e)

		if errors.As(e, &matConditionError) {
			return nil, wrapAsConditionError(e, &ConditionErrorHint{ExplanatoryVars: newExplanatoryVarHints(bm, coeffsVIFs)})
		}

		return nil, e
	}

	sInv := new(mat.Dense)
	sInv.Mul(sRInv, sQ.T())

	// 偏回帰係数の標準誤差
	coeffsStandardErrors := make([]float64, bm.numOfExplanatoryVars)
	for i := range coeffsStandardErrors {
		coeffsStandardErrors[i] = standardError * math.Sqrt(sInv.At(i, i))
	}

	// 定数（y切片）の標準誤差
	interceptStandardError := func() float64 {
		var sum float64
		for i := 0; i < bm.numOfExplanatoryVars; i++ {
			for j := 0; j < bm.numOfExplanatoryVars; j++ {
				sum += bm.meansOfExplanatoryVars[i] * bm.meansOfExplanatoryVars[j] * sInv.At(i, j)
			}
		}
		return math.Sqrt((1/float64(bm.numOfObservations) + sum)) * standardError
	}()

	// 標準化偏回帰係数 β 及び t値
	coeffsStandardized, coeffsTStats := make([]float64, bm.numOfExplanatoryVars), make([]float64, bm.numOfExplanatoryVars)
	for i, coeff := range bm.coeffs {
		coeffsStandardized[i] = coeff * bm.standardDeviationOfExplanatoryVars[i] / bm.standardDeviationOfObjectiveVars
		coeffsTStats[i] = coeff / coeffsStandardErrors[i]
	}

	// 定数（y切片）のt値
	interceptTStat := bm.intercept / interceptStandardError

	pValues := func() []float64 {
		tDistribution := distuv.StudentsT{
			Mu:    0,
			Sigma: 1,
			Nu:    residualDegreeOfFreedom,
		}
		pvs := make([]float64, bm.numOfExplanatoryVars)
		for i, tstat := range coeffsTStats {
			pvs[i] = tDistribution.Survival(math.Abs(tstat)) * 2
		}
		return pvs
	}()

	// 目的変数と説明変数を一つの行列にまとめ、変数間の相関行列を求める
	allDense := mat.NewDense(bm.numOfObservations, bm.numOfExplanatoryVars+1, nil)
	allDense.SetCol(0, r.objectiveVars)
	row := 1
	for idx, explanatoryVars := range r.explanatoryVarsMatrix {
		if _, ok := r.disregardingExplanatoryVarsSet[idx]; ok {
			continue
		}
		allDense.SetCol(row, explanatoryVars)
		row++
	}
	corrDense := new(mat.SymDense)
	stat.CorrelationMatrix(corrDense, allDense, nil)

	// 目的変数と各説明変数の相関係数を取り出す
	coeffsCorrelations := make([]float64, bm.numOfExplanatoryVars)
	for i := range coeffsCorrelations {
		coeffsCorrelations[i] = corrDense.At(0, i+1)
	}

	// 相関行列から偏相関行列を求める
	corrDenseInv := new(mat.Dense)
	if err := corrDenseInv.Inverse(corrDense); err != nil {
		e := fmt.Errorf("cannot inverse a matrix(corrDense): %w", err)

		logger.Err.Println(e)

		if errors.As(e, &matConditionError) {
			return nil, wrapAsConditionError(e, &ConditionErrorHint{ExplanatoryVars: newExplanatoryVarHints(bm, coeffsVIFs)})
		}

		return nil, e
	}
	coeffsPartialCorrelations := make([]float64, bm.numOfExplanatoryVars)
	for i := range coeffsPartialCorrelations {
		coeffsPartialCorrelations[i] = -1 * corrDenseInv.At(0, i+1) / math.Sqrt(corrDenseInv.At(0, 0)*corrDenseInv.At(i+1, i+1))
	}

	// 偏相関行列から部分相関行列を求める
	coeffsPartCorrelations := func() []float64 {
		if bm.numOfExplanatoryVars < 2 {
			return coeffsPartialCorrelations
		}
		corrs, row := make([]float64, bm.numOfExplanatoryVars), 0
		for idx := range r.explanatoryVarsMatrix {
			if _, ok := r.disregardingExplanatoryVarsSet[idx]; ok {
				continue
			}
			r.disregardingExplanatoryVarsSet[idx] = struct{}{}
			_bm, err := (&Regression{
				objectiveVars:                  r.objectiveVars,
				explanatoryVarsMatrix:          r.explanatoryVarsMatrix,
				disregardingExplanatoryVarsSet: r.disregardingExplanatoryVarsSet,
			}).run()
			if err != nil {
				panic(err) // Error should never happens
			}
			delete(r.disregardingExplanatoryVarsSet, idx)
			corrs[row] = coeffsPartialCorrelations[row] * math.Sqrt(1-_bm.r2)
			row++
		}
		return corrs
	}()

	logger.Info.Printf("Completed: Number of explanatory variables = %d", bm.numOfExplanatoryVars)

	return newModel(&rawModel{
		basicRawModel:             bm,
		adjustedR2:                adjustedR2,
		standardError:             standardError,
		regressionSumOfSquares:    regressionSumOfSquares,
		regressionDegreeOfFreedom: int(regressionDegreeOfFreedom),
		regressionMeanOfSquares:   regressionSumOfSquares / regressionDegreeOfFreedom,
		regressionFstat:           regressionFstat,
		regressionProb:            regressionProb,
		residualSumOfSquares:      residualSumOfSquares,
		residualDegreeOfFreedom:   int(residualDegreeOfFreedom),
		residualMeanOfSquares:     residualSumOfSquares / residualDegreeOfFreedom,
		totalSumOfSquares:         regressionSumOfSquares + residualSumOfSquares,
		totalDegreeOfFreedom:      int(totalDegreeOfFreedom),
		interceptStandardError:    interceptStandardError,
		coeffsStandardErrors:      coeffsStandardErrors,
		coeffsStandardized:        coeffsStandardized,
		interceptTStat:            interceptTStat,
		coeffsTStats:              coeffsTStats,
		coeffsProbs:               pValues,
		coeffsCorrelations:        coeffsCorrelations,
		coeffsPartialCorrelations: coeffsPartialCorrelations,
		coeffsPartCorrelations:    coeffsPartCorrelations,
		coeffsTolerances:          coeffsTolerances,
		coeffsVIFs:                coeffsVIFs,
	}), nil
}

// ValidateExplanatoryVars returns indexes of invalid explanatory variables.
// It considers an explanatory variable is not valid if is has all same observed values.
func (r *Regression) ValidateExplanatoryVars() []int {
	var invalidExplanatoryVarIndexes []int

EACH_EXPVAR:
	for i := range r.explanatoryVarsMatrix {
		if _, ok := r.disregardingExplanatoryVarsSet[i]; ok {
			continue
		}

		if len(r.explanatoryVarsMatrix[i]) == 0 {
			continue
		}

		for j := range r.explanatoryVarsMatrix[i][1:] {
			if r.explanatoryVarsMatrix[i][0] != r.explanatoryVarsMatrix[i][j+1] {
				continue EACH_EXPVAR
			}
		}

		invalidExplanatoryVarIndexes = append(invalidExplanatoryVarIndexes, i)
	}

	return invalidExplanatoryVarIndexes
}

// BackwardElimination : 変数減少法で解析する
//
// `forcedExpVarsIndexesSet`に指定したインデックスの説明変数は強制投入（必ず使用）される
func (r *Regression) BackwardElimination(p float64, forcedExpVarsIndexesSet map[int]struct{}) ([]Model, error) {
	originalDisregardingExplanatoryVarsSet := make(map[int]struct{}, len(r.disregardingExplanatoryVarsSet))
	for k, v := range r.disregardingExplanatoryVarsSet {
		originalDisregardingExplanatoryVarsSet[k] = v
	}
	defer func() {
		r.disregardingExplanatoryVarsSet = originalDisregardingExplanatoryVarsSet
	}()

	var models []Model
	for {
		model, err := r.Run()
		if err != nil {
			return models, err
		}
		models = append(models, *model)
		if model.NumOfExplanatoryVars == 1 {
			break
		}
		eliminationTarget, border := (*ExplanatoryVarResult)(nil), p
		for i := range model.ExplanatoryVars {
			// 強制投入する変数は除かない
			if _, ok := forcedExpVarsIndexesSet[model.ExplanatoryVars[i].OriginalIndex]; ok {
				continue
			}
			if model.ExplanatoryVars[i].Prob > border {
				eliminationTarget = &model.ExplanatoryVars[i]
				border = model.ExplanatoryVars[i].Prob
			}
		}
		if eliminationTarget == nil {
			break
		}
		logger.Info.Printf("Eliminate %s having p-value = %f", eliminationTarget.Label, eliminationTarget.Prob)
		r.DisregardIndex(eliminationTarget.OriginalIndex)
	}
	return models, nil
}
