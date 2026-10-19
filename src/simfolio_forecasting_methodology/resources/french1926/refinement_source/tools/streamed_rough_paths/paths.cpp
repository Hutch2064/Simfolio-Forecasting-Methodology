#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numpy/random/distributions.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style>;
using Dot = double(*)(int*, double*, int*, double*, int*);

Array multiplier(Array phis, Array weights, Array root, Array initial,
                 Array means, Array variances, py::object capsule, uintptr_t dot_address) {
    int n = phis.size(), stride = 1;
    auto* generator = static_cast<bitgen_t*>(PyCapsule_GetPointer(capsule.ptr(), "BitGenerator"));
    if (!generator) throw py::error_already_set();
    const double *phi = phis.data(), *w = weights.data(), *r = root.data();
    const double *mu = means.data(), *v = variances.data();
    for (int i=0;i<n;i++) for (int j=0;j<n;j++)
        if (i!=j && r[i*n+j]!=0.) throw py::value_error("streamed path kernel requires diagonal root");
    std::vector<double> state(initial.data(), initial.data()+n);
    Array result(means.size());
    auto* out = result.mutable_data();
    auto dot = reinterpret_cast<Dot>(dot_address);
    {
        py::gil_scoped_release release;
        for (py::ssize_t t=0;t<means.size();t++) {
            double h = dot(&n,const_cast<double*>(w),&stride,state.data(),&stride);
            out[t] = std::exp(.5*(h-mu[t])-.25*v[t]);
            for (int i=0;i<n;i++) {
                // NumPy normal(0,1) uses this exact distribution and draw order.
                // Consume zero-loading and final-step draws as in the reference.
                double z = 0.+1.*random_standard_normal(generator);
                double innovation = r[i*n+i]!=0. ? r[i*n+i]*z : 0.;
                state[i] = phi[i]*state[i]+innovation;
            }
        }
    }
    return result;
}

void map_path(Array phis, Array weights, Array root, Array initial,
              Array means, Array variances, py::object capsule, uintptr_t dot_address,
              Array return_mean, Array base_path, py::array_t<double,0> output) {
    int n=phis.size(),stride=1;
    auto* generator=static_cast<bitgen_t*>(PyCapsule_GetPointer(capsule.ptr(),"BitGenerator"));
    if(!generator) throw py::error_already_set();
    const double *phi=phis.data(),*w=weights.data(),*r=root.data();
    const double *mu=means.data(),*v=variances.data(),*mean=return_mean.data(),*base=base_path.data();
    for(int i=0;i<n;i++) for(int j=0;j<n;j++)
        if(i!=j && r[i*n+j]!=0.) throw py::value_error("streamed path kernel requires diagonal root");
    std::vector<double> state(initial.data(),initial.data()+n);
    auto* out=output.mutable_data();
    auto output_stride=output.strides(0)/sizeof(double);
    auto dot=reinterpret_cast<Dot>(dot_address);
    {
        py::gil_scoped_release release;
        for(py::ssize_t t=0;t<means.size();t++) {
            double h=dot(&n,const_cast<double*>(w),&stride,state.data(),&stride);
            double scale=std::exp(.5*(h-mu[t])-.25*v[t]);
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

PYBIND11_MODULE(rough_streamed_native, m) {
    m.def("multiplier", &multiplier);
    m.def("map_path", &map_path);
}
