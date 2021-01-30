#include "myopt.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstdio>
#include "def.h"
using namespace std;
using namespace Eigen;
myopt::myopt(myopt::Algorithm a, size_t dim)
    : _algo(a), _dim(dim), _cond(_default_stop_cond()), _data(nullptr), _solver(nullptr)
{
}
StopCond myopt::_default_stop_cond()
{
    StopCond sc;
    sc.stop_val = -1 * INF;
    sc.xtol_rel = 1e-15;
    sc.ftol_rel = 1e-15;
    sc.gtol     = 1e-15;
    sc.history  = 2;// only two element
    sc.max_iter = 100;
    sc.max_eval = 300;
    return sc;
}

myopt::Result myopt::optimize(Eigen::VectorXd& x0, double& y)
{
    switch (_algo)
    { 
        case CG:
            _solver=new class CG(_func, _dim, _cond, _data);
            break;
        case BFGS:
            _solver=new class BFGS(_func, _dim, _cond, _data);
            break;
        default:
            cerr<<"unsurpport algorithm:"<<_algo<<endl;
            return  INVALID_ARGS ;
    }
    _solver->set_param(_params);
    return _solver->minimize(x0, y);

}
myopt::~myopt()
{
    if (_solver != nullptr ) delete _solver;
}
void myopt::set_stop_val(double v) {_cond.stop_val=v;}
void myopt::set_algo_param(const std::map<std::string, double>& p){_params=p;}
void myopt::set_xtol_rel(double v){_cond.xtol_rel=v;}
void myopt::set_ftol_rel(double v){_cond.ftol_rel=v;}
void myopt::set_gtol(double v){_cond.gtol=v;}
void myopt::set_history(size_t h){_cond.history=h+1;}
void myopt::set_max_eval(size_t v){_cond.max_eval=v;}
void myopt::set_max_iter(size_t v){_cond.max_iter=v;}  // max line search
void myopt::set_min_objective(ObjFunc f, void* data){_func=f;_data=data;}
size_t myopt::get_dimension() const noexcept{return _dim;}
std::string myopt::get_algorithm_name() const noexcept
{
    #define C(A)  case A: return #A 
        switch (_algo)
        {
            C(CG);
            C(BFGS);
            default:
                cerr <<"undefine algorithm:"<<to_string(_algo)<<endl;
        }     
    #undef C   
}
std::string myopt::explain_result(Result r)
{
#define C(A) case A: return #A
    switch(r)
    {
        C(FAILURE         );
        C(INVALID_ARGS    );
        C(INVALID_INITIAL );
        C(NANINF          );
        C(SUCCESS         );
        C(STOPVAL_REACHED);
        C(FTOL_REACHED);
        C(XTOL_REACHED);
        C(GTOL_REACHED);
        C(MAXEVAL_REACHED);
        C(MAXITER_REACHED);
        default:
            return "Unknown reason" + to_string(r);
    }
#undef C
}

Solver::Solver(ObjFunc f, size_t dim, StopCond sc, void* d)
    : _func(f),
     // _line_search_trial(1.0),
      _eval_counter(0),
      _iter_counter(0),
      _dim(dim),
      _cond(sc),
      _data(d),
      _result(myopt::SUCCESS), 
      _history_x(queue<VectorXd>()),
      _history_y(queue<double>())
     // _c_decrease(0.01), 
     //  _c_curvature(0.9)
{}
Solver::~Solver() {}
double Solver::_run_func(const Eigen::VectorXd& xs, Eigen::VectorXd& g, bool need_g)
{
    double val=_func(xs,g,need_g,_data);
    ++_eval_counter;
    if (_besty>val){
        _besty=val;
        _bestx=xs;
    }
    return val;
}
void Solver::set_param(const std::map<std::string, double>& param){_params=param;}
void Solver::_init()
{
    _eval_counter = 0;
    _iter_counter = 0;
    while (!_history_x.empty()){
        _history_x.pop();
        _history_y.pop();
    }
    _besty=INF;
    _current_y=INF;
    _bestx=VectorXd::Constant(_dim, 1, INF);
    _current_x=VectorXd::Constant(_dim, 1, INF);
    _current_g=VectorXd::Constant(_dim, 1, INF);
}
myopt::Result Solver::minimize(Eigen::VectorXd& x0, double& y)
{
    _init();
    _current_x=x0;
    _current_y=_run_func(x0,_current_g,true);
    while (!_limit_reached())
    {
        _one_iter();
        ++_iter_counter;
        _update_hist();
    }
    x0=_bestx;
    y =_besty;
    return _result;

}
bool Solver::_limit_reached()
{
    if (_result==myopt::SUCCESS)
    {
        if(_besty<_cond.stop_val) _result=myopt::STOPVAL_REACHED;
        else if (_history_x.size()>_cond.history &&(_history_x.front()-_history_x.back()).norm()<_cond.xtol_rel)
           _result=myopt::XTOL_REACHED;
        else if(_history_y.size()>_cond.history &&fabs(_history_y.front()-_history_y.back())<_cond.ftol_rel)
            _result=myopt::FTOL_REACHED;
        else if(_current_g.norm()<_cond.gtol) _result=myopt::GTOL_REACHED;
        else if(_eval_counter>_cond.max_eval) _result=myopt::MAXEVAL_REACHED;
        else if(_iter_counter>_cond.max_iter) _result=myopt::MAXITER_REACHED;
    }
     return _result != myopt::SUCCESS;
}
void Solver::_update_hist()
{
    assert(_history_x.size()==_history_y.size());
    _history_x.push(_current_x);
    _history_y.push(_current_y);
    while (_history_x.size()>_cond.history){
        _history_x.pop();
        _history_y.pop();
    }
}
void CG::_init()
{
    Solver::_init();//虚函数，派生类和基类的同名函数就不同了。
    _former_g=VectorXd(_dim);
    //_former_y=VectorXd(_dim);
    _former_direction=VectorXd(_dim);
    c1=0.05;c2=0.1;
}
double CG::_beta_FR() const noexcept
{
return _current_g.squaredNorm()/_former_g.squaredNorm();
}
double CG::_beta_PRP() const noexcept
{
    return _current_g.dot(_current_g - _former_g) / _former_g.squaredNorm();
}
double  CG::_beta_SW() const noexcept
{
    return _current_g.dot(_current_g-_former_g)/(_former_direction.dot(_current_g-_former_g));
}
 myopt::Result CG::_one_iter()
 {
     const size_t inner_count=_iter_counter%_dim;
     double lambda=0;
     double y=0;
     VectorXd x(_dim);
     VectorXd g(_dim);
     VectorXd direction(_dim);

     if(inner_count==0) direction=-_current_g;
     else{
         double beta=_beta_FR();
         direction=-_current_g+beta*_former_direction;
     }
     size_t  maxsearch=20;
     _line_search_bin(direction,lambda,x,g,y,maxsearch);
     _former_direction=direction;
     _former_g=_current_g;
     //_former_y
    _current_g=g;
    _current_x=x;
    _current_y=y;
    return myopt::SUCCESS;
 }
void BFGS::_init()
{
    Solver::_init();
    _invB=MatrixXd::Identity(_dim ,_dim);
}
myopt::Result BFGS::_one_iter()
{
    VectorXd direction(_dim);
    VectorXd x(_dim);
    VectorXd g(_dim);
    double y=0;
    double lambda=0;
    size_t maxsearch=20;
    direction=-1*_invB*_current_g;
    _line_search_bin(direction,lambda,x,g,y,maxsearch);
    VectorXd sk=lambda*direction;
    VectorXd yk=g-_current_g;
    _invB=_invB+(1/sk.dot(yk)+(yk.transpose()*_invB*yk)/pow(sk.dot(yk),2))*sk*sk.transpose()-
           1/sk.dot(yk)*(_invB*yk*sk.transpose()+sk*yk.transpose()*_invB);
    _current_g=g;
    _current_x=x;
    _current_y=y;
    return myopt::SUCCESS;
}
bool Solver::_line_search_bin(const Eigen::VectorXd& direction, double& lambda, Eigen::VectorXd& x,
                                  Eigen::VectorXd& g, double& y, size_t max_search)
{   // binary search
    const double epslon=1e-4;
    g=_current_g;
    x=_current_x;
    y=_current_y;
    double f=INF;
    double lambda1=0;
    double lambda2=100/g.norm();
    const Eigen::VectorXd x0=x;
    Eigen::VectorXd x1(_dim);
    Eigen::VectorXd x2(_dim);
    Eigen::VectorXd g1(_dim);
    Eigen::VectorXd g2(_dim);
    size_t in_iter =0;
    while (f>epslon && in_iter<max_search)
    { 
        double f1,f2;
        lambda=(lambda1+lambda2)/2;
        x=x0+lambda*direction;
        x1=x0+lambda1*direction;
        x2=x0+lambda2*direction;
        _run_func(x1,g1,true);
        f1=g1.dot(direction);
         _run_func(x2,g2,true);
        f2=g2.dot(direction);
         _run_func(x,g,true);
        f=g.dot(direction);
        if(f*f1<0){lambda2=lambda;}
        else if(f*f2<0){lambda1=lambda;}
        else 
            cerr<<"bin search error"<<endl ;
        in_iter++;

    }
    return true ;
}