#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numpy/random/distributions.h>
#include <cmath>
#include <vector>
#include <cstdint>
namespace py=pybind11;
using Array=py::array_t<double,py::array::c_style>;
using Dot=double (*)(int*,double*,int*,double*,int*);
void map_path(Array phis,Array weights,Array root,Array initial,Array means,Array variances,
              py::object capsule,uintptr_t dot_address,Array return_mean,Array baseline,
              py::array_t<double,0> output,Array memory_phis,Array coefficients,double sd,
              Array memory_initial,double level,Array memory_mean,Array memory_variance,
              py::object memory_capsule,bool enabled,Array level_response,double level_delta) {
    auto* rng=static_cast<bitgen_t*>(PyCapsule_GetPointer(capsule.ptr(),"BitGenerator"));
    auto* memory_rng=static_cast<bitgen_t*>(PyCapsule_GetPointer(memory_capsule.ptr(),"BitGenerator"));
    if(!rng || !memory_rng) throw py::error_already_set();
    int n=phis.size(),k=memory_phis.size(),stride=1;
    if(weights.size()!=n || root.size()!=n*n || initial.size()!=n ||
       means.size()!=variances.size() || baseline.size()!=means.size() || return_mean.size()!=means.size() ||
       output.size()!=means.size() || coefficients.size()!=k+1 || memory_initial.size()!=k+1 ||
       memory_mean.size()!=means.size() || memory_variance.size()!=means.size() || level_response.size()!=means.size())
        throw py::value_error("coupled memory dimensions");
    Dot dot=reinterpret_cast<Dot>(dot_address);
    const double *phi=phis.data(),*w=weights.data(),*r=root.data(),*mu=means.data(),*v=variances.data();
    const double *mean=return_mean.data(),*base=baseline.data(),*mp=memory_phis.data(),*c=coefficients.data();
    const double *mm=memory_mean.data(),*mv=memory_variance.data(),*lg=level_response.data();
    std::vector<double> state(initial.data(),initial.data()+n),memory(memory_initial.data(),memory_initial.data()+k+1);
    auto* out=output.mutable_data();auto output_stride=output.strides(0)/sizeof(double);
    py::gil_scoped_release release;
    for(py::ssize_t t=0;t<means.size();t++) {
        double h=dot(&n,const_cast<double*>(w),&stride,state.data(),&stride);
        double multiplier=std::exp(.5*(h-mu[t])-.25*v[t]);
        if(enabled) {
            double next=0.;for(int j=0;j<k+1;j++)next+=c[j]*memory[j];
            next+=sd*random_standard_normal(memory_rng);
            for(int j=0;j<k;j++)memory[j+1]=mp[j]*memory[j+1]+(1.-mp[j])*next;
            memory[0]=next;
            double conventional=level+next-mm[t];
            if(level_delta!=0.)conventional+=level_delta*lg[t];
            multiplier*=std::exp(.5*conventional-.25*mv[t]);
        }
        double value=mean[t]+(base[t]-mean[t])*multiplier;
        out[t*output_stride]=value < -1. ? -1. : value > 1. ? 1. : value;
        for(int i=0;i<n;i++) {
            double z=0.+1.*random_standard_normal(rng);
            double innovation=r[i*n+i]!=0. ? r[i*n+i]*z : 0.;
            state[i]=phi[i]*state[i]+innovation;
        }
    }
}
PYBIND11_MODULE(volatility_level_native,m){m.def("map_path",&map_path);}
