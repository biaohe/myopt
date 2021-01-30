#pragma once
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>
#include <queue>
typedef std::function<double(const Eigen::VectorXd& input, Eigen::VectorXd& grad, bool need_g, void* data)> ObjFunc;
class Solver;
struct StopCond
{
    double stop_val;  // when the fom is below _stop_val, stop optimization
    double xtol_rel;
    double ftol_rel;
    double gtol;
    size_t history;  // history to remember, for delta based stop condition (xtol and ftol)
    size_t max_iter;
    size_t max_eval;
};
class myopt
{
public:
    enum Algorithm
    {
        CG = 0,
        BFGS,
       
        // MBFGS, // Google: superlinear nonconvex for papers
        // NUM_ALGORITHMS  // number of algorithm
    };
    enum Result
    {  // copied from
        FAILURE         = -1,
        INVALID_ARGS    = -2,
        INVALID_INITIAL = -3,  // starting point is NAN|INF
        NANINF          = -4,           // for those algorithms that can't recover from inf|nan
        SUCCESS         = 0,
        STOPVAL_REACHED,
        FTOL_REACHED,
        XTOL_REACHED,
        GTOL_REACHED,
        MAXEVAL_REACHED,
        MAXITER_REACHED, 
    };
    myopt(Algorithm, size_t);
    Result optimize(Eigen::VectorXd& x0, double& y);
    ~myopt();

    
    void set_stop_val(double);
    void set_algo_param(const std::map<std::string, double>&);
    void set_xtol_rel(double);
    void set_ftol_rel(double);
    void set_gtol(double);
    void set_history(size_t);
    void set_max_eval(size_t);
    void set_max_iter(size_t);  // max line search
    void set_min_objective(ObjFunc, void* data);
    size_t get_dimension() const noexcept;
    std::string get_algorithm_name() const noexcept;
    static std::string explain_result(Result);

private:
    const Algorithm _algo;
    const size_t    _dim;

    StopCond  _cond;
    void*     _data;
    Solver*   _solver;
    ObjFunc   _func;
    std::map<std::string, double> _params;
    StopCond  _default_stop_cond();
};

// Abstract class for solver
class Solver
{
public:
    Solver(ObjFunc, size_t, StopCond, void* data);
    virtual void set_param(const std::map<std::string, double>& param);
    virtual ~Solver();
    virtual myopt::Result minimize(Eigen::VectorXd& x0, double& y);
    //virtual myopt::Result minimize(Eigen::VectorXd& x0, double& y)=0;
private:
    ObjFunc _func;
    
protected:
    size_t _eval_counter;
    size_t _iter_counter;

    size_t        _dim;
    StopCond      _cond;
    void*         _data;
    myopt::Result _result;
    std::queue<Eigen::VectorXd> _history_x;
    std::queue<double>          _history_y;
   
    std::map<std::string, double> _params;
    Eigen::VectorXd _bestx;
    double _besty;

    // updated by _one_iter()
    Eigen::VectorXd _current_x;
    Eigen::VectorXd _current_g;
    double _current_y;

    virtual void _init();  // clear counter, best_x, best_y, set params
    virtual void _update_hist();
   // virtual void _set_linesearch_factor(double c1, double c2);
    virtual bool _line_search_bin(const Eigen::VectorXd& direction, double& alpha, Eigen::VectorXd& x,
                                  Eigen::VectorXd& g, double& y, size_t max_search);
                                    // binary search
    virtual bool _exat_line_search(const Eigen::VectorXd& direction, double& alpha, Eigen::VectorXd& x,
                                  Eigen::VectorXd& g, double& y, size_t max_search);
    virtual bool _limit_reached();  // return SUCCESS if not to stop
    virtual double _run_func(const Eigen::VectorXd& xs, Eigen::VectorXd& g, bool need_g);
    virtual myopt::Result _one_iter() = 0;
   // virtual void _set_trial(double);
   // virtual double _get_trial() const noexcept;
};

class CG : public Solver
{
    double c1, c2;  // param to control line search
    void _init();
    Eigen::VectorXd _former_g;
    Eigen::VectorXd _former_direction;
    myopt::Result _one_iter();
    double _beta_FR() const noexcept; // FLETCHER-REEVES update
    double _beta_PRP() const noexcept; // POLAK-RIBIERE update
    double _beta_SW() const noexcept; // Sorenson and Wolfe
public:
    using Solver::Solver;
};

class BFGS : public Solver
{
    Eigen::MatrixXd _invB;
    void _init();
    myopt::Result _one_iter();
public:
    using Solver::Solver;
};


