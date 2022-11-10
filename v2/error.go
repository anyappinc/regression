package regression

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

var (
	ErrNearSingular    = &ConditionError{isExactlySingular: false}
	ErrExactlySingular = &ConditionError{isExactlySingular: true}

	matConditionError    = mat.Condition(0)           // matrix singular or near-singular
	matConditionErrorInf = mat.Condition(math.Inf(1)) // matrix exactly singular
)

type ConditionError struct {
	err               error
	isExactlySingular bool
	Hint              *ConditionErrorHint
}

func (e ConditionError) Error() string {
	if e.err == nil {
		return ""
	}
	return e.err.Error()
}

func (e ConditionError) Is(err error) bool {
	if condErr, ok := err.(*ConditionError); ok {
		return e.isExactlySingular == condErr.isExactlySingular
	}
	return false
}

func (e ConditionError) Unwrap() error {
	return e.err
}

func wrapAsConditionError(err error, hint *ConditionErrorHint) *ConditionError {
	return &ConditionError{
		err:               err,
		isExactlySingular: errors.Is(err, matConditionErrorInf),
		Hint:              hint,
	}
}

type ConditionErrorHint struct {
	ExplanatoryVars []ExplanatoryVarHint
}

type ExplanatoryVarHint struct {
	OriginalIndex int     // 元々のインデックス
	Label         string  // 名称
	Coeff         float64 // 偏回帰係数 B
	VIF           float64 // 共線性の統計量 VIF
}

func newExplanatoryVarHints(basicRawModel *basicRawModel, vifs []float64) []ExplanatoryVarHint {
	hints := make([]ExplanatoryVarHint, basicRawModel.numOfExplanatoryVars)

	for i := 0; i < basicRawModel.numOfExplanatoryVars; i++ {
		hints[i] = ExplanatoryVarHint{
			OriginalIndex: basicRawModel.indexesTable[i],
			Label:         basicRawModel.explanatoryVarsLabels[i],
			Coeff:         basicRawModel.coeffs[i],
			VIF:           vifs[i],
		}
	}

	return hints
}
