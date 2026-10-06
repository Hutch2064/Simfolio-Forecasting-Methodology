#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numpy/random/distributions.h>
#include <cmath>
#include <vector>
#include <stdexcept>
namespace py=pybind11;
using Array=py::array_t<double,py::array::c_style>;
void map_path(Array phis,Array weights,Array rough_sd,Array initial,Array common,
              Array independent,double level,Array means,Array uniforms,Array nodes,
              py::object capsule,py::array_t<double,0> output,Array baseline_sd,
              Array normalizer_mean,Array normalizer_variance,bool normalize) {
    auto* generator=static_cast<bitgen_t*>(PyCapsule_GetPointer(capsule.ptr(),"BitGenerator"));
    if(!generator) throw py::error_already_set();
    int n=phis.size(),m=common.size(),k=nodes.size();
    if(weights.size()!=n || initial.size()!=n || rough_sd.size()!=n-m || independent.size()!=m ||
       means.size()!=uniforms.size() || output.size()!=means.size() || k<2)
        throw py::value_error("joint volatility array dimensions");
    if(normalize && (baseline_sd.size()!=means.size() || normalizer_mean.size()!=means.size() || normalizer_variance.size()!=means.size()))
        throw py::value_error("joint normalizer array dimensions");
    const double *phi=phis.data(),*w=weights.data(),*r=rough_sd.data(),*c=common.data(),*v=independent.data();
    const double *mu=means.data(),*u=uniforms.data(),*z=nodes.data();
    const double *reference=baseline_sd.data(),*nm=normalizer_mean.data(),*nv=normalizer_variance.data();
    std::vector<double> state(initial.data(),initial.data()+n);
    auto* out=output.mutable_data();auto stride=output.strides(0)/sizeof(double);
    py::gil_scoped_release release;
    for(py::ssize_t t=0;t<means.size();t++) {
        double a=random_standard_normal(generator),b=random_standard_normal(generator);
        for(int i=0;i<m;i++)state[i]=phi[i]*state[i]+c[i]*a+v[i]*b;
        for(int i=m;i<n;i++)state[i]=phi[i]*state[i]+r[i-m]*random_standard_normal(generator);
        double h=level;for(int i=0;i<n;i++)h+=w[i]*state[i];
        double sd=normalize ? reference[t]*std::exp(.5*(h-nm[t])-.25*nv[t]) : std::exp(.5*h)/100.;
        if(!std::isfinite(sd)) throw std::runtime_error("nonfinite joint volatility forecast scale");
        double position=std::fmin(std::fmax(u[t],0.),1.)*double(k-1);
        int lo=int(std::floor(position)),hi=std::min(lo+1,k-1);double fraction=position-lo;
        double left=std::fmin(std::fmax(mu[t]+sd*z[lo],-1.),1.);
        double right=std::fmin(std::fmax(mu[t]+sd*z[hi],-1.),1.);
        out[t*stride]=left*(1.-fraction)+right*fraction;
    }
}
PYBIND11_MODULE(joint_gaussian_volatility_native,m){m.def("map_path",&map_path);}
