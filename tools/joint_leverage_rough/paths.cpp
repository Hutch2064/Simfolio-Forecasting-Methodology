#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numpy/random/distributions.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style>;
using Dot = double(*)(int*, double*, int*, double*, int*);

void map_path(Array phis, Array weights, Array root, Array initial,
              Array means, Array variances, py::object capsule, uintptr_t dot_address,
              Array return_mean, Array base_path, py::array_t<double,0> output,
              Array ms_phis, Array ms_loading, Array ms_common, Array ms_independent,
              Array ms_initial, double ms_level, Array ms_means, Array ms_variances,
              py::object ms_capsule, bool enabled, double rho, Array gaussian_scores, double observed_score) {
    int n=phis.size(),stride=1;
    auto* generator=static_cast<bitgen_t*>(PyCapsule_GetPointer(capsule.ptr(),"BitGenerator"));
    if(!generator) throw py::error_already_set();
    const double *phi=phis.data(),*w=weights.data(),*r=root.data();
    const double *mu=means.data(),*v=variances.data(),*mean=return_mean.data(),*base=base_path.data();
    for(int i=0;i<n;i++) for(int j=0;j<n;j++)
        if(i!=j && r[i*n+j]!=0.) throw py::value_error("streamed path kernel requires diagonal root");
    std::vector<double> state(initial.data(),initial.data()+n);
    int m=ms_phis.size();
    auto* ms_generator=static_cast<bitgen_t*>(PyCapsule_GetPointer(ms_capsule.ptr(),"BitGenerator"));
    if(!ms_generator) throw py::error_already_set();
    const double *mp=ms_phis.data(), *ml=ms_loading.data(), *mc=ms_common.data(), *mi=ms_independent.data();
    const double *mm=ms_means.data(), *mv=ms_variances.data();
    std::vector<double> ms_state(ms_initial.data(),ms_initial.data()+m);
    const double* return_scores=gaussian_scores.data();
    double aggregate_common=0.,aggregate_independent=0.;
    for(int i=0;i<m;i++) { aggregate_common+=ml[i]*mc[i];aggregate_independent+=ml[i]*mi[i]; }
    double aggregate_sd=std::hypot(aggregate_common,aggregate_independent);
    double a0=aggregate_sd>0.?aggregate_common/aggregate_sd:0.;
    double a1=aggregate_sd>0.?aggregate_independent/aggregate_sd:0.;
    double shrink=std::sqrt(1-rho*rho)-1.;
    auto* out=output.mutable_data();
    auto output_stride=output.strides(0)/sizeof(double);
    auto dot=reinterpret_cast<Dot>(dot_address);
    {
        py::gil_scoped_release release;
        for(py::ssize_t t=0;t<means.size();t++) {
            double h=dot(&n,const_cast<double*>(w),&stride,state.data(),&stride);
            double scale=std::exp(.5*(h-mu[t])-.25*v[t]);
            if(enabled) {
                double z0=random_standard_normal(ms_generator),z1=random_standard_normal(ms_generator);
                if(rho!=0.) {
                    double previous_score=t==0 ? observed_score : return_scores[t-1];
                    double change=rho*previous_score+shrink*(a0*z0+a1*z1);
                    z0+=a0*change;z1+=a1*change;
                }
                for(int i=0;i<m;i++) ms_state[i]=mp[i]*ms_state[i]+mc[i]*z0+mi[i]*z1;
                double ms_h=ms_level;
                for(int i=0;i<m;i++) ms_h+=ml[i]*ms_state[i];
                scale*=std::exp(.5*(ms_h-mm[t])-.25*mv[t]);
            }
            double value=mean[t]+(base[t]-mean[t])*scale;
            out[t*output_stride]=value < -1. ? -1. : value > 1. ? 1. : value;
            for(int i=0;i<n;i++) {
                double z=0.+1.*random_standard_normal(generator);
                double innovation=r[i*n+i]!=0. ? r[i*n+i]*z : 0.;
                state[i]=phi[i]*state[i]+innovation;
            }
        }
    }
}

PYBIND11_MODULE(joint_leverage_native, m) {
    m.def("map_path", &map_path);
}
