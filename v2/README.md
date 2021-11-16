regression v2
=======

[go-reference-badge]: https://pkg.go.dev/badge/github.com/anyappinc/regression/v2.svg
[go-reference-url]: https://pkg.go.dev/github.com/anyappinc/regression/v2
[license-image]: http://img.shields.io/badge/license-MIT-green.svg?style=flat-square
[license-url]: LICENSE

[![Go Reference][go-reference-badge]][go-reference-url]
[![License][license-image]][license-url]

Multiple linear regression analysis library written in Go 

*NOTE: This project is originally based on https://github.com/sajari/regression*

installation
------------

    $ go get github.com/anyappinc/regression/v2

example usage
-------------

Import the package, create a regression and add data to it. You can use as many variables as you like, in the below example there are 3 variables for each observation.

```go
package main

import (
	"fmt"

	"github.com/anyappinc/regression/v2"
)

func main() {
	r := regression.NewRegression()
	r.SetObjectiveVariableLabel("Murders per annum per 1,000,000 inhabitants")
	r.SetExplanatoryVariableLabel(0, "Inhabitants")
	r.SetExplanatoryVariableLabel(1, "Percent with incomes below $5000")
	r.SetExplanatoryVariableLabel(2, "Percent unemployed")
	r.AddObservations(
		regression.NewObservation(11.2, []float64{587000, 16.5, 6.2}),
		regression.NewObservation(13.4, []float64{643000, 20.5, 6.4}),
		regression.NewObservation(40.7, []float64{635000, 26.3, 9.3}),
		regression.NewObservation(5.3, []float64{692000, 16.5, 5.3}),
		regression.NewObservation(24.8, []float64{1248000, 19.2, 7.3}),
		regression.NewObservation(12.7, []float64{643000, 16.5, 5.9}),
		regression.NewObservation(20.9, []float64{1964000, 20.2, 6.4}),
		regression.NewObservation(35.7, []float64{1531000, 21.3, 7.6}),
		regression.NewObservation(8.7, []float64{713000, 17.2, 4.9}),
		regression.NewObservation(9.6, []float64{749000, 14.3, 6.4}),
		regression.NewObservation(14.5, []float64{7895000, 18.1, 6}),
		regression.NewObservation(26.9, []float64{762000, 23.1, 7.4}),
		regression.NewObservation(15.7, []float64{2793000, 19.1, 5.8}),
		regression.NewObservation(36.2, []float64{741000, 24.7, 8.6}),
		regression.NewObservation(18.1, []float64{625000, 18.6, 6.5}),
		regression.NewObservation(28.9, []float64{854000, 24.9, 8.3}),
		regression.NewObservation(14.9, []float64{716000, 17.9, 6.7}),
		regression.NewObservation(25.8, []float64{921000, 22.4, 8.6}),
		regression.NewObservation(21.7, []float64{595000, 20.2, 8.4}),
		regression.NewObservation(25.7, []float64{3353000, 16.9, 6.7}),
	)
	model, err := r.Run()
	if err != nil {
		log.Fatalln(err)
	}

	fmt.Printf("Regression formula:\n%v\n", model.FormulaString())
}
```

Note: You can also add observations one by one.

Once calculated, you can look at the R^2, Standard Error, ANOVA, Coefficients, etc. e.g.

```go
// Get the coefficient for the "Inhabitants" variable 0:
c := model.ExplanatoryVars[0].Coeff
```

You can also use the model to predict new observation

```go
prediction, err := model.Predict([]float64{587000, 16.5, 6.2})
```
