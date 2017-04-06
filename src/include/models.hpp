
#ifndef MODELS_HPP
#define MODELS_HPP
#define STAN__SERVICES__COMMAND_HPP
#include <rstan/rstaninc.hpp>
// Code generated by Stan version 2.14

#include <stan/model/model_header.hpp>

namespace model_bacon_namespace {

using std::istream;
using std::string;
using std::stringstream;
using std::vector;
using stan::io::dump;
using stan::math::lgamma;
using stan::model::prob_grad;
using namespace stan::math;

typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_d;
typedef Eigen::Matrix<double,1,Eigen::Dynamic> row_vector_d;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> matrix_d;

static int current_statement_begin__;

class model_bacon : public prob_grad {
private:
    int N;
    int K;
    int nu;
    vector_d depth;
    vector_d obs_age;
    vector_d obs_err;
    vector_d c_depth_bottom;
    vector_d c_depth_top;
    vector<int> which_c;
    double delta_c;
    double acc_mean;
    double acc_var;
    double mem_mean;
    double mem_strength;
    double acc_alpha;
    double acc_beta;
    double mem_alpha;
    double mem_beta;
public:
    model_bacon(stan::io::var_context& context__,
        std::ostream* pstream__ = 0)
        : prob_grad(0) {
        typedef boost::ecuyer1988 rng_t;
        rng_t base_rng(0);  // 0 seed default
        ctor_body(context__, base_rng, pstream__);
    }

    template <class RNG>
    model_bacon(stan::io::var_context& context__,
        RNG& base_rng__,
        std::ostream* pstream__ = 0)
        : prob_grad(0) {
        ctor_body(context__, base_rng__, pstream__);
    }

    template <class RNG>
    void ctor_body(stan::io::var_context& context__,
                   RNG& base_rng__,
                   std::ostream* pstream__) {
        current_statement_begin__ = -1;

        static const char* function__ = "model_bacon_namespace::model_bacon";
        (void) function__; // dummy call to supress warning
        size_t pos__;
        (void) pos__; // dummy call to supress warning
        std::vector<int> vals_i__;
        std::vector<double> vals_r__;
        double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // suppress unused var warning

        // initialize member variables
        context__.validate_dims("data initialization", "N", "int", context__.to_vec());
        N = int(0);
        vals_i__ = context__.vals_i("N");
        pos__ = 0;
        N = vals_i__[pos__++];
        context__.validate_dims("data initialization", "K", "int", context__.to_vec());
        K = int(0);
        vals_i__ = context__.vals_i("K");
        pos__ = 0;
        K = vals_i__[pos__++];
        context__.validate_dims("data initialization", "nu", "int", context__.to_vec());
        nu = int(0);
        vals_i__ = context__.vals_i("nu");
        pos__ = 0;
        nu = vals_i__[pos__++];
        validate_non_negative_index("depth", "N", N);
        depth = vector_d(static_cast<Eigen::VectorXd::Index>(N));
        context__.validate_dims("data initialization", "depth", "vector_d", context__.to_vec(N));
        vals_r__ = context__.vals_r("depth");
        pos__ = 0;
        size_t depth_i_vec_lim__ = N;
        for (size_t i_vec__ = 0; i_vec__ < depth_i_vec_lim__; ++i_vec__) {
            depth[i_vec__] = vals_r__[pos__++];
        }
        validate_non_negative_index("obs_age", "N", N);
        obs_age = vector_d(static_cast<Eigen::VectorXd::Index>(N));
        context__.validate_dims("data initialization", "obs_age", "vector_d", context__.to_vec(N));
        vals_r__ = context__.vals_r("obs_age");
        pos__ = 0;
        size_t obs_age_i_vec_lim__ = N;
        for (size_t i_vec__ = 0; i_vec__ < obs_age_i_vec_lim__; ++i_vec__) {
            obs_age[i_vec__] = vals_r__[pos__++];
        }
        validate_non_negative_index("obs_err", "N", N);
        obs_err = vector_d(static_cast<Eigen::VectorXd::Index>(N));
        context__.validate_dims("data initialization", "obs_err", "vector_d", context__.to_vec(N));
        vals_r__ = context__.vals_r("obs_err");
        pos__ = 0;
        size_t obs_err_i_vec_lim__ = N;
        for (size_t i_vec__ = 0; i_vec__ < obs_err_i_vec_lim__; ++i_vec__) {
            obs_err[i_vec__] = vals_r__[pos__++];
        }
        validate_non_negative_index("c_depth_bottom", "K", K);
        c_depth_bottom = vector_d(static_cast<Eigen::VectorXd::Index>(K));
        context__.validate_dims("data initialization", "c_depth_bottom", "vector_d", context__.to_vec(K));
        vals_r__ = context__.vals_r("c_depth_bottom");
        pos__ = 0;
        size_t c_depth_bottom_i_vec_lim__ = K;
        for (size_t i_vec__ = 0; i_vec__ < c_depth_bottom_i_vec_lim__; ++i_vec__) {
            c_depth_bottom[i_vec__] = vals_r__[pos__++];
        }
        validate_non_negative_index("c_depth_top", "K", K);
        c_depth_top = vector_d(static_cast<Eigen::VectorXd::Index>(K));
        context__.validate_dims("data initialization", "c_depth_top", "vector_d", context__.to_vec(K));
        vals_r__ = context__.vals_r("c_depth_top");
        pos__ = 0;
        size_t c_depth_top_i_vec_lim__ = K;
        for (size_t i_vec__ = 0; i_vec__ < c_depth_top_i_vec_lim__; ++i_vec__) {
            c_depth_top[i_vec__] = vals_r__[pos__++];
        }
        context__.validate_dims("data initialization", "which_c", "int", context__.to_vec(N));
        validate_non_negative_index("which_c", "N", N);
        which_c = std::vector<int>(N,int(0));
        vals_i__ = context__.vals_i("which_c");
        pos__ = 0;
        size_t which_c_limit_0__ = N;
        for (size_t i_0__ = 0; i_0__ < which_c_limit_0__; ++i_0__) {
            which_c[i_0__] = vals_i__[pos__++];
        }
        context__.validate_dims("data initialization", "delta_c", "double", context__.to_vec());
        delta_c = double(0);
        vals_r__ = context__.vals_r("delta_c");
        pos__ = 0;
        delta_c = vals_r__[pos__++];
        context__.validate_dims("data initialization", "acc_mean", "double", context__.to_vec());
        acc_mean = double(0);
        vals_r__ = context__.vals_r("acc_mean");
        pos__ = 0;
        acc_mean = vals_r__[pos__++];
        context__.validate_dims("data initialization", "acc_var", "double", context__.to_vec());
        acc_var = double(0);
        vals_r__ = context__.vals_r("acc_var");
        pos__ = 0;
        acc_var = vals_r__[pos__++];
        context__.validate_dims("data initialization", "mem_mean", "double", context__.to_vec());
        mem_mean = double(0);
        vals_r__ = context__.vals_r("mem_mean");
        pos__ = 0;
        mem_mean = vals_r__[pos__++];
        context__.validate_dims("data initialization", "mem_strength", "double", context__.to_vec());
        mem_strength = double(0);
        vals_r__ = context__.vals_r("mem_strength");
        pos__ = 0;
        mem_strength = vals_r__[pos__++];

        // validate, data variables
        check_greater_or_equal(function__,"N",N,0);
        check_greater_or_equal(function__,"K",K,0);
        check_greater_or_equal(function__,"nu",nu,0);
        check_greater_or_equal(function__,"delta_c",delta_c,0);
        check_greater_or_equal(function__,"acc_mean",acc_mean,0);
        check_greater_or_equal(function__,"acc_var",acc_var,0);
        check_greater_or_equal(function__,"mem_mean",mem_mean,0);
        check_greater_or_equal(function__,"mem_strength",mem_strength,0);
        // initialize data variables
        acc_alpha = double(0);
        stan::math::fill(acc_alpha,DUMMY_VAR__);
        acc_beta = double(0);
        stan::math::fill(acc_beta,DUMMY_VAR__);
        mem_alpha = double(0);
        stan::math::fill(mem_alpha,DUMMY_VAR__);
        mem_beta = double(0);
        stan::math::fill(mem_beta,DUMMY_VAR__);

        try {
            stan::math::assign(acc_alpha, (pow(acc_mean,2) / acc_var));
            stan::math::assign(acc_beta, (acc_mean / acc_var));
            stan::math::assign(mem_alpha, (mem_strength * mem_mean));
            stan::math::assign(mem_beta, (mem_strength * (1 - mem_mean)));
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e,current_statement_begin__);
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }

        // validate transformed data
        check_greater_or_equal(function__,"acc_alpha",acc_alpha,0);
        check_greater_or_equal(function__,"acc_beta",acc_beta,0);
        check_greater_or_equal(function__,"mem_alpha",mem_alpha,0);
        check_greater_or_equal(function__,"mem_beta",mem_beta,0);

        // set parameter ranges
        num_params_r__ = 0U;
        param_ranges_i__.clear();
        ++num_params_r__;
        num_params_r__ += K;
        ++num_params_r__;
    }

    ~model_bacon() { }


    void transform_inits(const stan::io::var_context& context__,
                         std::vector<int>& params_i__,
                         std::vector<double>& params_r__,
                         std::ostream* pstream__) const {
        stan::io::writer<double> writer__(params_r__,params_i__);
        size_t pos__;
        (void) pos__; // dummy call to supress warning
        std::vector<double> vals_r__;
        std::vector<int> vals_i__;

        if (!(context__.contains_r("R")))
            throw std::runtime_error("variable R missing");
        vals_r__ = context__.vals_r("R");
        pos__ = 0U;
        context__.validate_dims("initialization", "R", "double", context__.to_vec());
        // generate_declaration R
        double R(0);
        R = vals_r__[pos__++];
        try {
            writer__.scalar_lub_unconstrain(0,1,R);
        } catch (const std::exception& e) { 
            throw std::runtime_error(std::string("Error transforming variable R: ") + e.what());
        }

        if (!(context__.contains_r("alpha")))
            throw std::runtime_error("variable alpha missing");
        vals_r__ = context__.vals_r("alpha");
        pos__ = 0U;
        context__.validate_dims("initialization", "alpha", "vector_d", context__.to_vec(K));
        // generate_declaration alpha
        vector_d alpha(static_cast<Eigen::VectorXd::Index>(K));
        for (int j1__ = 0U; j1__ < K; ++j1__)
            alpha(j1__) = vals_r__[pos__++];
        try {
            writer__.vector_lb_unconstrain(0,alpha);
        } catch (const std::exception& e) { 
            throw std::runtime_error(std::string("Error transforming variable alpha: ") + e.what());
        }

        if (!(context__.contains_r("age0")))
            throw std::runtime_error("variable age0 missing");
        vals_r__ = context__.vals_r("age0");
        pos__ = 0U;
        context__.validate_dims("initialization", "age0", "double", context__.to_vec());
        // generate_declaration age0
        double age0(0);
        age0 = vals_r__[pos__++];
        try {
            writer__.scalar_lb_unconstrain(0,age0);
        } catch (const std::exception& e) { 
            throw std::runtime_error(std::string("Error transforming variable age0: ") + e.what());
        }

        params_r__ = writer__.data_r();
        params_i__ = writer__.data_i();
    }

    void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                         std::ostream* pstream__) const {
      std::vector<double> params_r_vec;
      std::vector<int> params_i_vec;
      transform_inits(context, params_i_vec, params_r_vec, pstream__);
      params_r.resize(params_r_vec.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r(i) = params_r_vec[i];
    }


    template <bool propto__, bool jacobian__, typename T__>
    T__ log_prob(vector<T__>& params_r__,
                 vector<int>& params_i__,
                 std::ostream* pstream__ = 0) const {

        T__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // suppress unused var warning

        T__ lp__(0.0);
        stan::math::accumulator<T__> lp_accum__;

        // model parameters
        stan::io::reader<T__> in__(params_r__,params_i__);

        T__ R;
        (void) R;  // dummy to suppress unused var warning
        if (jacobian__)
            R = in__.scalar_lub_constrain(0,1,lp__);
        else
            R = in__.scalar_lub_constrain(0,1);

        Eigen::Matrix<T__,Eigen::Dynamic,1>  alpha;
        (void) alpha;  // dummy to suppress unused var warning
        if (jacobian__)
            alpha = in__.vector_lb_constrain(0,K,lp__);
        else
            alpha = in__.vector_lb_constrain(0,K);

        T__ age0;
        (void) age0;  // dummy to suppress unused var warning
        if (jacobian__)
            age0 = in__.scalar_lb_constrain(0,lp__);
        else
            age0 = in__.scalar_lb_constrain(0);


        // transformed parameters
        T__ w;
        (void) w;  // dummy to suppress unused var warning
        stan::math::initialize(w, DUMMY_VAR__);
        stan::math::fill(w,DUMMY_VAR__);
        Eigen::Matrix<T__,Eigen::Dynamic,1>  x(static_cast<Eigen::VectorXd::Index>(K));
        (void) x;  // dummy to suppress unused var warning
        stan::math::initialize(x, DUMMY_VAR__);
        stan::math::fill(x,DUMMY_VAR__);
        Eigen::Matrix<T__,Eigen::Dynamic,1>  c_ages(static_cast<Eigen::VectorXd::Index>((K + 1)));
        (void) c_ages;  // dummy to suppress unused var warning
        stan::math::initialize(c_ages, DUMMY_VAR__);
        stan::math::fill(c_ages,DUMMY_VAR__);
        Eigen::Matrix<T__,Eigen::Dynamic,1>  Mod_age(static_cast<Eigen::VectorXd::Index>(N));
        (void) Mod_age;  // dummy to suppress unused var warning
        stan::math::initialize(Mod_age, DUMMY_VAR__);
        stan::math::fill(Mod_age,DUMMY_VAR__);


        try {
            stan::math::assign(w, pow(R,(delta_c / 1)));
            stan::math::assign(get_base1_lhs(x,1,"x",1), get_base1(alpha,1,"alpha",1));
            for (int i = 2; i <= K; ++i) {

                stan::math::assign(get_base1_lhs(x,i,"x",1), ((w * get_base1(x,(i - 1),"x",1)) + ((1 - w) * get_base1(alpha,i,"alpha",1))));
            }
            stan::math::assign(get_base1_lhs(c_ages,1,"c_ages",1), age0);
            stan::model::assign(c_ages, 
                        stan::model::cons_list(stan::model::index_min_max(2, (K + 1)), stan::model::nil_index_list()), 
                        add(age0,cumulative_sum(multiply(alpha,delta_c))), 
                        "assigning variable c_ages");
            stan::math::assign(Mod_age, add(stan::model::rvalue(c_ages, stan::model::cons_list(stan::model::index_multi(which_c), stan::model::nil_index_list()), "c_ages"),elt_multiply(stan::model::rvalue(alpha, stan::model::cons_list(stan::model::index_multi(which_c), stan::model::nil_index_list()), "alpha"),subtract(depth,stan::model::rvalue(c_depth_top, stan::model::cons_list(stan::model::index_multi(which_c), stan::model::nil_index_list()), "c_depth_top")))));
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e,current_statement_begin__);
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }

        // validate transformed parameters
        if (stan::math::is_uninitialized(w)) {
            std::stringstream msg__;
            msg__ << "Undefined transformed parameter: w";
            throw std::runtime_error(msg__.str());
        }
        for (int i0__ = 0; i0__ < K; ++i0__) {
            if (stan::math::is_uninitialized(x(i0__))) {
                std::stringstream msg__;
                msg__ << "Undefined transformed parameter: x" << '[' << i0__ << ']';
                throw std::runtime_error(msg__.str());
            }
        }
        for (int i0__ = 0; i0__ < (K + 1); ++i0__) {
            if (stan::math::is_uninitialized(c_ages(i0__))) {
                std::stringstream msg__;
                msg__ << "Undefined transformed parameter: c_ages" << '[' << i0__ << ']';
                throw std::runtime_error(msg__.str());
            }
        }
        for (int i0__ = 0; i0__ < N; ++i0__) {
            if (stan::math::is_uninitialized(Mod_age(i0__))) {
                std::stringstream msg__;
                msg__ << "Undefined transformed parameter: Mod_age" << '[' << i0__ << ']';
                throw std::runtime_error(msg__.str());
            }
        }

        const char* function__ = "validate transformed params";
        (void) function__;  // dummy to suppress unused var warning
        check_greater_or_equal(function__,"w",w,0);
        check_less_or_equal(function__,"w",w,1);

        // model body
        try {

            lp_accum__.add(normal_log<propto__>(age0, get_base1(obs_age,1,"obs_age",1), 100));
            lp_accum__.add(beta_log<propto__>(R, mem_alpha, mem_beta));
            lp_accum__.add(gamma_log<propto__>(alpha, acc_alpha, acc_beta));
            lp_accum__.add(student_t_log<propto__>(obs_age, nu, Mod_age, obs_err));
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e,current_statement_begin__);
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }

        lp_accum__.add(lp__);
        return lp_accum__.sum();

    } // log_prob()

    template <bool propto, bool jacobian, typename T_>
    T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
               std::ostream* pstream = 0) const {
      std::vector<T_> vec_params_r;
      vec_params_r.reserve(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        vec_params_r.push_back(params_r(i));
      std::vector<int> vec_params_i;
      return log_prob<propto,jacobian,T_>(vec_params_r, vec_params_i, pstream);
    }


    void get_param_names(std::vector<std::string>& names__) const {
        names__.resize(0);
        names__.push_back("R");
        names__.push_back("alpha");
        names__.push_back("age0");
        names__.push_back("w");
        names__.push_back("x");
        names__.push_back("c_ages");
        names__.push_back("Mod_age");
    }


    void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
        dimss__.resize(0);
        std::vector<size_t> dims__;
        dims__.resize(0);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dims__.push_back(K);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dims__.push_back(K);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dims__.push_back((K + 1));
        dimss__.push_back(dims__);
        dims__.resize(0);
        dims__.push_back(N);
        dimss__.push_back(dims__);
    }

    template <typename RNG>
    void write_array(RNG& base_rng__,
                     std::vector<double>& params_r__,
                     std::vector<int>& params_i__,
                     std::vector<double>& vars__,
                     bool include_tparams__ = true,
                     bool include_gqs__ = true,
                     std::ostream* pstream__ = 0) const {
        vars__.resize(0);
        stan::io::reader<double> in__(params_r__,params_i__);
        static const char* function__ = "model_bacon_namespace::write_array";
        (void) function__; // dummy call to supress warning
        // read-transform, write parameters
        double R = in__.scalar_lub_constrain(0,1);
        vector_d alpha = in__.vector_lb_constrain(0,K);
        double age0 = in__.scalar_lb_constrain(0);
        vars__.push_back(R);
        for (int k_0__ = 0; k_0__ < K; ++k_0__) {
            vars__.push_back(alpha[k_0__]);
        }
        vars__.push_back(age0);

        if (!include_tparams__) return;
        // declare and define transformed parameters
        double lp__ = 0.0;
        (void) lp__; // dummy call to supress warning
        stan::math::accumulator<double> lp_accum__;

        double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // suppress unused var warning

        double w(0.0);
        (void) w;  // dummy to suppress unused var warning
        stan::math::initialize(w, std::numeric_limits<double>::quiet_NaN());
        stan::math::fill(w,DUMMY_VAR__);
        vector_d x(static_cast<Eigen::VectorXd::Index>(K));
        (void) x;  // dummy to suppress unused var warning
        stan::math::initialize(x, std::numeric_limits<double>::quiet_NaN());
        stan::math::fill(x,DUMMY_VAR__);
        vector_d c_ages(static_cast<Eigen::VectorXd::Index>((K + 1)));
        (void) c_ages;  // dummy to suppress unused var warning
        stan::math::initialize(c_ages, std::numeric_limits<double>::quiet_NaN());
        stan::math::fill(c_ages,DUMMY_VAR__);
        vector_d Mod_age(static_cast<Eigen::VectorXd::Index>(N));
        (void) Mod_age;  // dummy to suppress unused var warning
        stan::math::initialize(Mod_age, std::numeric_limits<double>::quiet_NaN());
        stan::math::fill(Mod_age,DUMMY_VAR__);


        try {
            stan::math::assign(w, pow(R,(delta_c / 1)));
            stan::math::assign(get_base1_lhs(x,1,"x",1), get_base1(alpha,1,"alpha",1));
            for (int i = 2; i <= K; ++i) {

                stan::math::assign(get_base1_lhs(x,i,"x",1), ((w * get_base1(x,(i - 1),"x",1)) + ((1 - w) * get_base1(alpha,i,"alpha",1))));
            }
            stan::math::assign(get_base1_lhs(c_ages,1,"c_ages",1), age0);
            stan::model::assign(c_ages, 
                        stan::model::cons_list(stan::model::index_min_max(2, (K + 1)), stan::model::nil_index_list()), 
                        add(age0,cumulative_sum(multiply(alpha,delta_c))), 
                        "assigning variable c_ages");
            stan::math::assign(Mod_age, add(stan::model::rvalue(c_ages, stan::model::cons_list(stan::model::index_multi(which_c), stan::model::nil_index_list()), "c_ages"),elt_multiply(stan::model::rvalue(alpha, stan::model::cons_list(stan::model::index_multi(which_c), stan::model::nil_index_list()), "alpha"),subtract(depth,stan::model::rvalue(c_depth_top, stan::model::cons_list(stan::model::index_multi(which_c), stan::model::nil_index_list()), "c_depth_top")))));
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e,current_statement_begin__);
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }

        // validate transformed parameters
        check_greater_or_equal(function__,"w",w,0);
        check_less_or_equal(function__,"w",w,1);

        // write transformed parameters
        vars__.push_back(w);
        for (int k_0__ = 0; k_0__ < K; ++k_0__) {
            vars__.push_back(x[k_0__]);
        }
        for (int k_0__ = 0; k_0__ < (K + 1); ++k_0__) {
            vars__.push_back(c_ages[k_0__]);
        }
        for (int k_0__ = 0; k_0__ < N; ++k_0__) {
            vars__.push_back(Mod_age[k_0__]);
        }

        if (!include_gqs__) return;
        // declare and define generated quantities


        try {
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e,current_statement_begin__);
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }

        // validate generated quantities

        // write generated quantities
    }

    template <typename RNG>
    void write_array(RNG& base_rng,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                     bool include_tparams = true,
                     bool include_gqs = true,
                     std::ostream* pstream = 0) const {
      std::vector<double> params_r_vec(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r_vec[i] = params_r(i);
      std::vector<double> vars_vec;
      std::vector<int> params_i_vec;
      write_array(base_rng,params_r_vec,params_i_vec,vars_vec,include_tparams,include_gqs,pstream);
      vars.resize(vars_vec.size());
      for (int i = 0; i < vars.size(); ++i)
        vars(i) = vars_vec[i];
    }

    static std::string model_name() {
        return "model_bacon";
    }


    void constrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
        std::stringstream param_name_stream__;
        param_name_stream__.str(std::string());
        param_name_stream__ << "R";
        param_names__.push_back(param_name_stream__.str());
        for (int k_0__ = 1; k_0__ <= K; ++k_0__) {
            param_name_stream__.str(std::string());
            param_name_stream__ << "alpha" << '.' << k_0__;
            param_names__.push_back(param_name_stream__.str());
        }
        param_name_stream__.str(std::string());
        param_name_stream__ << "age0";
        param_names__.push_back(param_name_stream__.str());

        if (!include_gqs__ && !include_tparams__) return;
        param_name_stream__.str(std::string());
        param_name_stream__ << "w";
        param_names__.push_back(param_name_stream__.str());
        for (int k_0__ = 1; k_0__ <= K; ++k_0__) {
            param_name_stream__.str(std::string());
            param_name_stream__ << "x" << '.' << k_0__;
            param_names__.push_back(param_name_stream__.str());
        }
        for (int k_0__ = 1; k_0__ <= (K + 1); ++k_0__) {
            param_name_stream__.str(std::string());
            param_name_stream__ << "c_ages" << '.' << k_0__;
            param_names__.push_back(param_name_stream__.str());
        }
        for (int k_0__ = 1; k_0__ <= N; ++k_0__) {
            param_name_stream__.str(std::string());
            param_name_stream__ << "Mod_age" << '.' << k_0__;
            param_names__.push_back(param_name_stream__.str());
        }

        if (!include_gqs__) return;
    }


    void unconstrained_param_names(std::vector<std::string>& param_names__,
                                   bool include_tparams__ = true,
                                   bool include_gqs__ = true) const {
        std::stringstream param_name_stream__;
        param_name_stream__.str(std::string());
        param_name_stream__ << "R";
        param_names__.push_back(param_name_stream__.str());
        for (int k_0__ = 1; k_0__ <= K; ++k_0__) {
            param_name_stream__.str(std::string());
            param_name_stream__ << "alpha" << '.' << k_0__;
            param_names__.push_back(param_name_stream__.str());
        }
        param_name_stream__.str(std::string());
        param_name_stream__ << "age0";
        param_names__.push_back(param_name_stream__.str());

        if (!include_gqs__ && !include_tparams__) return;
        param_name_stream__.str(std::string());
        param_name_stream__ << "w";
        param_names__.push_back(param_name_stream__.str());
        for (int k_0__ = 1; k_0__ <= K; ++k_0__) {
            param_name_stream__.str(std::string());
            param_name_stream__ << "x" << '.' << k_0__;
            param_names__.push_back(param_name_stream__.str());
        }
        for (int k_0__ = 1; k_0__ <= (K + 1); ++k_0__) {
            param_name_stream__.str(std::string());
            param_name_stream__ << "c_ages" << '.' << k_0__;
            param_names__.push_back(param_name_stream__.str());
        }
        for (int k_0__ = 1; k_0__ <= N; ++k_0__) {
            param_name_stream__.str(std::string());
            param_name_stream__ << "Mod_age" << '.' << k_0__;
            param_names__.push_back(param_name_stream__.str());
        }

        if (!include_gqs__) return;
    }

}; // model

} // namespace




#endif
