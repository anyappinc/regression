package regression

import (
	"fmt"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type basicRawModel struct {
	indexesTable                       []int            // 元のインデックスと実際のインデックスの対応表
	objectiveVarLabel                  string           // 目的変数の名称
	explanatoryVarsLabels              []string         // 各説明変数の名称
	objectiveVarsDense                 *mat.Dense       // 目的変数の観測値の行列
	explanatoryVarsDense               *mat.Dense       // 説明変数の観測値の行列
	numOfObservations                  int              // 分析に用いた観測値の数
	numOfExplanatoryVars               int              // 分析に用いた説明変数の数
	disregardedExplanatoryVarsSet      map[int]struct{} // 分析に使用されなかったインデックスのセット
	meanOfObjectiveVars                float64          // 目的変数の観測値の平均
	standardDeviationOfObjectiveVars   float64          // 目的変数の観測値の標準偏差
	meansOfExplanatoryVars             []float64        // 各説明変数の観測値の平均
	standardDeviationOfExplanatoryVars []float64        // 各説明変数の観測値の標準偏差
	coeffs                             []float64        // 偏回帰係数 B
	intercept                          float64          // y切片
	predictedVals                      []float64        // 予測値
	residuals                          []float64        // 残差
	totalVariation                     float64          // 全変動
	explainedVariation                 float64          // 回帰変動
	unexplainedVariation               float64          // 残差変動
	r                                  float64          // 重相関係数
	r2                                 float64          // 決定係数
}

type rawModel struct {
	*basicRawModel
	adjustedR2                float64   // 自由度調整済み決定係数
	standardError             float64   // 回帰の標準誤差（推定値の標準偏差）
	regressionSumOfSquares    float64   // 回帰の平方和
	regressionDegreeOfFreedom int       // 回帰の自由度
	regressionMeanOfSquares   float64   // 回帰の平均平方
	regressionFstat           float64   // 回帰のF値
	regressionProb            float64   // 回帰の有意確率
	residualSumOfSquares      float64   // 残差の平方和
	residualDegreeOfFreedom   int       // 残差の自由度
	residualMeanOfSquares     float64   // 残差の平均平方
	totalSumOfSquares         float64   // 合計の平方和
	totalDegreeOfFreedom      int       // 合計の自由度
	interceptStandardError    float64   // 定数（y切片）の標準誤差（非標準化係数 標準誤差）
	coeffsStandardErrors      []float64 // 偏回帰係数の標準誤差（非標準化係数 標準誤差）
	coeffsStandardized        []float64 // 標準化偏回帰係数 β
	interceptTStat            float64   // 定数（y切片）のt値
	coeffsTStats              []float64 // t値
	coeffsProbs               []float64 // 有意確率（p値）
	coeffsCorrelations        []float64 // ゼロ次相関（通常の相関係数）
	coeffsPartialCorrelations []float64 // 偏相関
	coeffsPartCorrelations    []float64 // 部分相関
	coeffsTolerances          []float64 // 共線性の統計量 許容度 TOL
	coeffsVIFs                []float64 // 共線性の統計量 VIF
}

// ANOVA : 分散分析の結果
type ANOVA struct {
	RegressionSumOfSquares    float64 // 回帰の平方和
	RegressionDegreeOfFreedom int     // 回帰の自由度
	RegressionMeanOfSquares   float64 // 回帰の平均平方
	RegressionFstat           float64 // 回帰のF値
	RegressionProb            float64 // 回帰の有意確率
	ResidualSumOfSquares      float64 // 残差の平方和
	ResidualDegreeOfFreedom   int     // 残差の自由度
	ResidualMeanOfSquares     float64 // 残差の平均平方
	TotalSumOfSquares         float64 // 合計の平方和
	TotalDegreeOfFreedom      int     // 合計の自由度
}

// InterceptResult : 回帰分析により算出された定数（y切片）の結果
type InterceptResult struct {
	Value         float64 // 定数（y切片）の値
	StandardError float64 // 標準誤差（非標準化係数 標準誤差）
	TStat         float64 // t値
}

// ExplanatoryVarResult : 回帰分析により算出された説明変数の結果
type ExplanatoryVarResult struct {
	OriginalIndex      int     // 元々のインデックス
	Label              string  // 名称
	Coeff              float64 // 偏回帰係数 B
	StandardError      float64 // 偏回帰係数の標準誤差（非標準化係数 標準誤差）
	StandardizedCoeff  float64 // 標準化偏回帰係数 β
	TStat              float64 // t値
	Prob               float64 // 有意確率（p値）
	Correlation        float64 // ゼロ次相関（通常の相関係数）
	PartialCorrelation float64 // 偏相関
	PartCorrelation    float64 // 部分相関
	Tolerance          float64 // 共線性の統計量 許容度 TOL
	VIF                float64 // 共線性の統計量 VIF
}

// ObservationsAnalysis : 観測値に関する分析結果
type ObservationsAnalysis struct {
	ObjectiveVarsDense                 *mat.Dense // 目的変数の観測値の行列
	ExplanatoryVarsDense               *mat.Dense // 説明変数の観測値の行列
	MeanOfObjectiveVars                float64    // 目的変数の観測値の平均
	StandardDeviationOfObjectiveVars   float64    // 目的変数の観測値の標準偏差
	MeansOfExplanatoryVars             []float64  // 各説明変数の観測値の平均
	StandardDeviationOfExplanatoryVars []float64  // 各説明変数の観測値の標準偏差
	PredictedVals                      []float64  // 予測値
	Residuals                          []float64  // 残差
}

// Model : 回帰モデル
type Model struct {
	observationsAnalysis          *ObservationsAnalysis  // 観測値に関する分析結果
	NumOfObservations             int                    // 分析に用いた観測値の数
	NumOfExplanatoryVars          int                    // 分析に用いた説明変数の数
	DisregardedExplanatoryVarsSet map[int]struct{}       // 分析に使用されなかったインデックスのセット
	UnexplainedVariation          float64                // 残差変動
	ExplainedVariation            float64                // 回帰変動
	TotalVariation                float64                // 全変動
	R                             float64                // 重相関係数
	R2                            float64                // 決定係数
	AdjustedR2                    float64                // 自由度調整済み決定係数
	StandardError                 float64                // 回帰の標準誤差（推定値の標準偏差）
	ANOVA                         *ANOVA                 // 分散分析
	ObjectiveVarLabel             string                 // 目的変数の名称
	Intercept                     *InterceptResult       // 定数（y切片）の分析結果
	ExplanatoryVars               []ExplanatoryVarResult // 各説明変数の分析結果
}

func newModel(rawModel *rawModel) *Model {
	explanatoryVars := make([]ExplanatoryVarResult, rawModel.numOfExplanatoryVars)
	for i := 0; i < rawModel.numOfExplanatoryVars; i++ {
		explanatoryVars[i] = ExplanatoryVarResult{
			OriginalIndex:      rawModel.indexesTable[i],
			Label:              rawModel.explanatoryVarsLabels[i],
			Coeff:              rawModel.coeffs[i],
			StandardError:      rawModel.coeffsStandardErrors[i],
			StandardizedCoeff:  rawModel.coeffsStandardized[i],
			TStat:              rawModel.coeffsTStats[i],
			Prob:               rawModel.coeffsProbs[i],
			Correlation:        rawModel.coeffsCorrelations[i],
			PartialCorrelation: rawModel.coeffsPartialCorrelations[i],
			PartCorrelation:    rawModel.coeffsPartCorrelations[i],
			Tolerance:          rawModel.coeffsTolerances[i],
			VIF:                rawModel.coeffsVIFs[i],
		}
	}

	return &Model{
		observationsAnalysis: &ObservationsAnalysis{
			ObjectiveVarsDense:                 rawModel.objectiveVarsDense,
			ExplanatoryVarsDense:               rawModel.explanatoryVarsDense,
			MeanOfObjectiveVars:                rawModel.meanOfObjectiveVars,
			StandardDeviationOfObjectiveVars:   rawModel.standardDeviationOfObjectiveVars,
			MeansOfExplanatoryVars:             rawModel.meansOfExplanatoryVars,
			StandardDeviationOfExplanatoryVars: rawModel.standardDeviationOfExplanatoryVars,
			PredictedVals:                      rawModel.predictedVals,
			Residuals:                          rawModel.residuals,
		},
		NumOfObservations:             rawModel.numOfObservations,
		NumOfExplanatoryVars:          rawModel.numOfExplanatoryVars,
		DisregardedExplanatoryVarsSet: rawModel.disregardedExplanatoryVarsSet,
		UnexplainedVariation:          rawModel.unexplainedVariation,
		ExplainedVariation:            rawModel.explainedVariation,
		TotalVariation:                rawModel.totalVariation,
		R:                             rawModel.r,
		R2:                            rawModel.r2,
		AdjustedR2:                    rawModel.adjustedR2,
		StandardError:                 rawModel.standardError,
		ANOVA: &ANOVA{
			RegressionSumOfSquares:    rawModel.regressionSumOfSquares,
			RegressionDegreeOfFreedom: rawModel.regressionDegreeOfFreedom,
			RegressionMeanOfSquares:   rawModel.regressionMeanOfSquares,
			RegressionFstat:           rawModel.regressionFstat,
			RegressionProb:            rawModel.regressionProb,
			ResidualSumOfSquares:      rawModel.residualSumOfSquares,
			ResidualDegreeOfFreedom:   rawModel.residualDegreeOfFreedom,
			ResidualMeanOfSquares:     rawModel.residualMeanOfSquares,
			TotalSumOfSquares:         rawModel.totalSumOfSquares,
			TotalDegreeOfFreedom:      rawModel.totalDegreeOfFreedom,
		},
		ObjectiveVarLabel: rawModel.objectiveVarLabel,
		Intercept: &InterceptResult{
			Value:         rawModel.intercept,
			StandardError: rawModel.interceptStandardError,
			TStat:         rawModel.interceptTStat,
		},
		ExplanatoryVars: explanatoryVars,
	}
}

func formatFloatForFormula(f float64) string {
	if f < 0 {
		return fmt.Sprintf(" - %.4f", -f)
	}
	return fmt.Sprintf(" + %.4f", f)
}

// FormulaString : 回帰モデル式を文字列で取得する
func (m *Model) FormulaString() string {
	formulaStrs := make([]string, len(m.ExplanatoryVars)*2)
	for i, ev := range m.ExplanatoryVars {
		formulaStrs[i*2] = formatFloatForFormula(ev.Coeff)
		formulaStrs[i*2+1] = "*" + ev.Label
	}
	return m.ObjectiveVarLabel + " =" + strings.Join(formulaStrs, "") + formatFloatForFormula(m.Intercept.Value)
}

// GetObservationsAnalysis : 観測値に関する分析結果を取得する
func (m *Model) GetObservationsAnalysis() *ObservationsAnalysis {
	return m.observationsAnalysis
}

// Predict calculates the predicted value
func (m *Model) Predict(vars []float64) (float64, error) {
	coeffs := make([]float64, m.NumOfExplanatoryVars)
	for i, ev := range m.ExplanatoryVars {
		coeffs[i] = ev.Coeff
	}
	return calcPredictedVal(vars, coeffs, m.Intercept.Value)
}
