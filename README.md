
<!-- README.md is generated from README.Rmd. Please edit that file -->

# hamstr: Hierarchical Accumulation Modelling with Stan and R <img src="man/figures/hex-hamstr.png" align="right" width = 120px/>

<!-- badges: start -->

[![codecov](https://codecov.io/gh/EarthSystemDiagnostics/hamstr/branch/dev/graph/badge.svg?token=gFBWomcqwc)](https://codecov.io/gh/EarthSystemDiagnostics/hamstr)
[![test-coverage](https://github.com/EarthSystemDiagnostics/hamstr/actions/workflows/test-coverage.yaml/badge.svg)](https://github.com/EarthSystemDiagnostics/hamstr/actions/workflows/test-coverage.yaml)
[![R-CMD-check](https://github.com/EarthSystemDiagnostics/hamstr/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/EarthSystemDiagnostics/hamstr/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

**hamstr** implements a *Bacon-like* (Blaauw and Christen, 2011)
sediment accumulation or age-depth model with hierarchically structured
multi-resolution sediment sections. The Bayesian model is implemented in
the Stan probabilistic programming language (<https://mc-stan.org/>).

## Installation

**hamstr** can be installed directly from Github

``` r
if (!require("remotes")) {
  install.packages("remotes")
}

remotes::install_github("earthsystemdiagnostics/hamstr", args = "--preclean", build_vignettes = FALSE)
```

## Using **hamstr**

Examples using the example core “MSB2K” from the
[rbacon](https://cran.r-project.org/web/packages/rbacon/index.html)
package.

``` r
library(hamstr)
library(rstan)

set.seed(20200827)
```

### Converting radiocarbon ages to calendar ages.

Unlike Bacon, **hamstr** does not do the conversion of radiocarbon dates
to calendar ages as part of the model fitting process. This must be done
in advance. **hamstr** includes the helper function `calibrate_14C_age`
to do this, which in turn uses the function `BchronCalibrate` from the
[Bchron](https://cran.r-project.org/web/packages/Bchron/index.html)
package.

Additionally, unlike Bacon, **hamstr** approximates the complex
empirical calendar age PDF that results from calibration into a single
point estimate and 1-sigma uncertainty. This is a necessary compromise
in order to be able to use the power of the Stan platform. Viewed in
context with the many other uncertainties in radiocarbon dates and the
resulting age-models this will not usually be a major issue.

The function `calibrate_14C_age` will append columns to a data.frame
with the calendar ages and 1-sigma uncertainties.

``` r
MSB2K_cal <- calibrate_14C_age(MSB2K, age.14C = "age", age.14C.se = "error")
```

The approximated calendar age PDFs can be compared with the empirical
PDFs with the function `compare_14C_PDF`

A sample of six dates are plotted here for the IntCal20 and Marine20
calibrations. This approximation is much less of an issue for marine
radiocarbon dates, as the cosmogenic radiocarbon signal has been
smoothed by mixing in the ocean.

``` r
i <- seq(1, 40, by = floor(40/6))[1:6]
compare_14C_PDF(MSB2K$age[i], MSB2K$error[i], cal_curve = "intcal20")+
  labs(title = "Intcal20")
```

<img src="man/figures/README-unnamed-chunk-4-1.svg" width="100%" />

``` r
compare_14C_PDF(MSB2K$age[i], MSB2K$error[i], cal_curve = "marine20") +
  labs(title = "Marine20")
```

<img src="man/figures/README-unnamed-chunk-5-1.svg" width="100%" />

### Fitting age-models with **hamstr**

Age-depth (sediment accumulation) models are fit with the function
`hamstr`. A vectors of depth, observed age and age uncertainty are
passed as arguments to the function.

``` r
hamstr_fit_1 <- hamstr(depth = MSB2K_cal$depth,
                       obs_age = MSB2K_cal$age.14C.cal,
                       obs_err = MSB2K_cal$age.14C.cal.se, 
                       # the seed argument for the sampler is set here so that
                       # this example always returns the same numerical result
                       stan_sampler_args = list(seed = 1))
```

The default plotting method shows the fitted age models together with
some diagnostic plots: a traceplot of the log-posterior to assess
convergence of the overall model; a plot of accumulation rate against
depth at each hierarchical level; the prior and posterior of the memory
parameter. By default the age-models are summarised to show the mean,
median, 25% and 95% posterior intervals. The data are shown as points
with their 1-sigma uncertainties. The structure of the sections is shown
along the top of the age-model plot.

``` r
plot(hamstr_fit_1)
```

<img src="man/figures/README-unnamed-chunk-7-1.svg" width="100%" />

A “spaghetti” plot can be created instead of shaded regions. This shows
a random sample of iterations from the posterior distribution
(realisation of the age-depth model). This can be slow if lots of
iterations are plotted, the default is to plot 1000 iterations.
Additionally, plotting of the diagnostic plots can be switched off.

``` r
plot(hamstr_fit_1, summarise = FALSE, plot_diagnostics = FALSE)
```

<img src="man/figures/README-unnamed-chunk-8-1.png" width="100%" />

#### Mean accumulation rate

There is no need to specify a prior value for the mean accumulation rate
(parameter `acc.mean` in Bacon) as in **hamstr**, this overall mean
accumulation rate is a full parameter estimated from the data.

By default, **hamstr** uses robust linear regression (`MASS::rlm`) to
estimate the mean accumulation rate from the data, and then uses this to
parametrise a prior distribution for the overall mean accumulation rate.
This prior is a half-normal with zero mean and standard deviation equal
to 10 times the estimated mean. Although this does introduce a slight
element of “double-dipping”, using the data twice (for both the prior
and likelihood), the resulting prior is only weakly-informative. The
advantage of this approach is that the prior is automatically scaled
appropriately regardless of the units of depth or age.

This prior can be checked visually against the posterior. The posterior
distribution should be much narrower than the weakly informative prior.

``` r
plot(hamstr_fit_1, type = "acc_mean_prior_post")
```

<img src="man/figures/README-unnamed-chunk-9-1.svg" width="100%" />

#### Other hyperparameters

Default parameter values for the shape of the gamma distributed
accumulation rates `acc_shape = 1.5`, the memory mean `mem_mean = 0.5`
and memory strength `mem_strength = 10`, are the same as for Bacon \>=
2.5.1.

### Setting the thickness, number, and hierarchical structure of the discrete sections

One of the more critical tuning parameters in the **Bacon** model is the
parameter `thick`, which determines the thickness and number of discrete
down-core sediment sections modelled. Finding a good or optimal value
for a given core is often critical to getting a good age-depth model.
Too few sections and the resulting age-model is very “blocky” and can
miss changes in sedimentation rate; however, counter-intuitively, too
many very thin sections can also often result in an age-model that
“under-fits” the data - a straight line through the age-control points
when a lower resolution model shows variation in accumulation rate.

The key structural difference between **Bacon** and **hamstr** models is
that with **hamstr** the sediment core is modelled at multiple
resolutions simultaneously with a hierarchical structure. This removes
the need to trade-off smoothness and flexibility.

The parameter `K` controls the number and structure of the hierarchical
sections. It is specified as a vector, where each value indicates the
number of new child sections for each parent section at each finer
hierarchical level. E.g. `c(10, 10)` would specify 10 sections at the
coarsest level, with 10 new sections at the next finer level for each
coarse section, giving a total of 100 sections at the highest / finest
resolution level. `c(10, 10, 10)` would specify 1000 sections at the
finest level and 3 hierarchical levels of 10, 100 and 1000 sections.

The structure is hierarchical in the sense that the modelled
accumulation rates for the parent sections act as priors for their child
sections; specifically, the mean accumulation rate for a given parent is
the mean of the gamma prior for its child sections. In turn, the overall
mean accumulation rate for the whole core is itself a parameter
estimated by the fitting process. The hierarchical structure of
increasing resolution allows the model to adapt to low-frequency changes
in the accumulation rate, that is changes between “regimes” of high or
low accumulation that persist for long periods.

By default `K` is chosen so that the number of hierarchical levels, and
the number of new child sections per level, are approximately equal,
e.g. c(4, 4, 4, 4). The total number of sections at the finest level is
set so that the resolution is 1 cm per section, up to a total length of
900 cm, above which the default remains 900 sections and a coarser
resolution is used. This can be changed from the default via the
parameter `K`.

For a given shape parameter `acc_shape`, increasing the number of
modelled hierarchical levels increases the total variance in the
accumulation rates at the highest / finest resolution level. From
**hamstr** version 0.5.0 and onwards, the total variance is controlled
by modifying the shape parameter according to the number of hierarchical
levels.

### Getting the fitted age models

The fitted age models can be obtained with the `predict` and `summary`
methods. *iter* is the iteration of the sampler, or “realisation” of the
age model.

``` r
predict(hamstr_fit_1)
#> # A tibble: 388,000 × 3
#>     iter depth   age
#>    <int> <dbl> <dbl>
#>  1     1  1.5  4463.
#>  2     1  2.52 4485.
#>  3     1  3.54 4512.
#>  4     1  4.56 4530.
#>  5     1  5.58 4541.
#>  6     1  6.60 4550.
#>  7     1  7.62 4559.
#>  8     1  8.65 4578.
#>  9     1  9.67 4598.
#> 10     1 10.7  4616.
#> # … with 387,990 more rows
```

`summary` returns the age model summarised over the realisations.

``` r
summary(hamstr_fit_1)
#> # A tibble: 97 × 13
#>    depth   idx par     mean se_mean    sd `2.5%` `25%` `50%` `75%` `97.5%` n_eff
#>    <dbl> <dbl> <chr>  <dbl>   <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl>   <dbl> <dbl>
#>  1  1.5      1 c_age… 4520.   1.64   64.5  4382. 4481. 4524. 4564.   4636. 1539.
#>  2  2.52     2 c_age… 4532.   1.52   60.4  4404. 4495. 4536. 4575.   4642. 1585.
#>  3  3.54     3 c_age… 4545.   1.40   56.8  4423. 4509. 4547. 4584.   4650. 1656.
#>  4  4.56     4 c_age… 4557.   1.28   53.7  4444. 4522. 4559. 4594.   4657. 1763.
#>  5  5.58     5 c_age… 4570.   1.17   51.0  4463. 4537. 4572. 4605.   4666. 1905.
#>  6  6.60     6 c_age… 4582.   1.07   48.8  4480. 4551. 4584. 4615.   4675. 2063.
#>  7  7.62     7 c_age… 4594.   0.987  47.0  4496. 4564. 4595. 4626.   4684. 2272.
#>  8  8.65     8 c_age… 4607.   0.906  45.0  4515. 4578. 4608. 4636.   4693. 2464.
#>  9  9.67     9 c_age… 4620.   0.839  43.0  4533. 4592. 4620. 4648.   4703. 2630.
#> 10 10.7     10 c_age… 4633.   0.790  41.7  4551. 4607. 4633. 4660.   4714. 2787.
#> # … with 87 more rows, and 1 more variable: Rhat <dbl>
```

The hierarchical structure of the sections makes it difficult to specify
the exact depth resolution that you want for your resulting age-depth
model. The `predict` method takes an additional argument `depth` to
interpolate to a specific set of depths. The function returns NA for
depths that are outside the modelled depths.

``` r
age.mods.interp <- predict(hamstr_fit_1, depth = seq(0, 100, by = 1))
```

These interpolated age models can summarised with the same function as
the original fitted objects, but the n_eff and Rhat information is lost.

``` r
summary(age.mods.interp)
#> # A tibble: 101 × 8
#>    depth  mean    sd `2.5%` `25%` `50%` `75%` `97.5%`
#>    <dbl> <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl>   <dbl>
#>  1     0  NaN   NA      NA    NA    NA    NA      NA 
#>  2     1  NaN   NA      NA    NA    NA    NA      NA 
#>  3     2 4526.  62.4  4394. 4488. 4530. 4569.   4639.
#>  4     3 4538.  58.6  4414. 4501. 4541. 4579.   4647.
#>  5     4 4550.  55.4  4433. 4515. 4553. 4588.   4654.
#>  6     5 4563.  52.5  4452. 4528. 4565. 4598.   4661.
#>  7     6 4575.  50.0  4471. 4542. 4577. 4609.   4669.
#>  8     7 4587.  48.0  4487. 4556. 4588. 4619.   4678.
#>  9     8 4599.  46.2  4503. 4570. 4600. 4630.   4687.
#> 10     9 4612.  44.2  4522. 4583. 4612. 4640.   4697.
#> # … with 91 more rows
```

### Getting and plotting the accumulation rate

The down-core accumulation rates are returned and plotted in both
depth-per-time, and time-per-depth units. If the input data are in years
and cm then the units will be cm/kyr and yrs/cm respectively. Note that
the acc_mean parameter in both **hamstr** and Bacon is parametrised in
terms of time per depth.

``` r
plot(hamstr_fit_1, type = "acc_rates")
#> Joining, by = "idx"
#> Joining, by = "depth"
```

<img src="man/figures/README-unnamed-chunk-14-1.svg" width="100%" />

``` r
summary(hamstr_fit_1, type = "acc_rates") 
#> Joining, by = "idx"
#> # A tibble: 192 × 13
#>    depth c_depth_top c_depth_bottom acc_rate_unit   idx   tau  mean    sd `2.5%`
#>    <dbl>       <dbl>          <dbl> <chr>         <dbl> <dbl> <dbl> <dbl>  <dbl>
#>  1  1.5         1.5            2.52 depth_per_ti…     1     0 123.   90.5   32.4
#>  2  2.52        2.52           3.54 depth_per_ti…     2     0 115.   78.5   34.2
#>  3  3.54        3.54           4.56 depth_per_ti…     3     0 113.   75.4   34.2
#>  4  4.56        4.56           5.58 depth_per_ti…     4     0 107.   64.9   36.6
#>  5  5.58        5.58           6.60 depth_per_ti…     5     0 108.   68.0   36.5
#>  6  6.60        6.60           7.62 depth_per_ti…     6     0 109.   69.5   36.6
#>  7  7.62        7.62           8.65 depth_per_ti…     7     0  99.7  55.5   38.3
#>  8  8.65        8.65           9.67 depth_per_ti…     8     0 101.   59.5   36.0
#>  9  9.67        9.67          10.7  depth_per_ti…     9     0 103.   65.3   34.9
#> 10 10.7        10.7           11.7  depth_per_ti…    10     0 100.   59.2   37.3
#> # … with 182 more rows, and 4 more variables: `25%` <dbl>, `50%` <dbl>,
#> #   `75%` <dbl>, `97.5%` <dbl>
```

### Diagnostic plots

Additional diagnostic plots are available. See ?plot.hamstr_fit for
options.

#### Plot modelled accumulation rates at each hierarchical level

``` r
plot(hamstr_fit_1, type = "hier_acc")
```

<img src="man/figures/README-unnamed-chunk-16-1.svg" width="100%" />

#### Plot memory prior and posterior

As for this example the highest resolution sections are approximately 1
cm thick, there is not much difference between R and w.

``` r
plot(hamstr_fit_1, type = "mem")
```

<img src="man/figures/README-unnamed-chunk-17-1.svg" width="100%" />

### Other `rstan` functions

Within the hamstr_fit object is an *rstan* object on which all the
standard rstan functions should operate correctly.

For example:

``` r
rstan::check_divergences(hamstr_fit_1$fit)
#> 0 of 4000 iterations ended with a divergence.

rstan::stan_rhat(hamstr_fit_1$fit)
#> `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="man/figures/README-unnamed-chunk-18-1.svg" width="100%" />

The first `alpha` parameter is the overall mean accumulation rate.

``` r
rstan::traceplot(hamstr_fit_1$fit, par = c("alpha[1]"),
                 inc_warmup = TRUE)
```

<img src="man/figures/README-unnamed-chunk-19-1.svg" width="100%" />

### References

-   Blaauw, Maarten, and J. Andrés Christen. 2011. Flexible Paleoclimate
    Age-Depth Models Using an Autoregressive Gamma Process. Bayesian
    Analysis 6 (3): 457-74. <doi:10.1214/ba/1339616472>.

-   Parnell, Andrew. 2016. Bchron: Radiocarbon Dating, Age-Depth
    Modelling, Relative Sea Level Rate Estimation, and Non-Parametric
    Phase Modelling. R package version 4.2.6.
    <https://CRAN.R-project.org/package=Bchron>

-   Stan Development Team (2020). RStan: the R interface to Stan. R
    package version 2.21.2. <http://mc-stan.org/>.
