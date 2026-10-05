#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstring>
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style>;
using Dot = double(*)(int*, double*, int*, double*, int*);
template<int FixedN>
py::tuple filter_fixed(Array observations, Array phis, Array weights, Array covariance, double level, uintptr_t dot_address) {
    const int n=FixedN?FixedN:phis.size();int stride=1,blas_n=n;
    const double *y=observations.data(), *phi=phis.data(), *w=weights.data(), *q=covariance.data();
    py::array_t<double> state(n), posterior({n,n});
    double *s=state.mutable_data(), *p=posterior.mutable_data();
    std::vector<double> predicted(n*n), pw(n), gain(n);
    double* prediction_matrix=predicted.data();
    auto dot=reinterpret_cast<Dot>(dot_address);
    double loglik=0., variance=0., log_variance=0.;
    bool fixed=false,ones=true;
    for(int i=0;i<n;i++)if(w[i]!=1.)ones=false;
    {
    py::gil_scoped_release release;
    for(int i=0;i<n;i++) {
        s[i]=0.;
        for(int j=0;j<n;j++) p[i*n+j]=q[i*n+j]/(1.-phi[i]*phi[j]);
    }
    for(py::ssize_t t=0;t<observations.size();t++) {
        if(!fixed) {
            for(int i=0;i<n;i++) {
                int j=0; double total=0.; auto fi=vdupq_n_f64(phi[i]);
                for(;j+1<n;j+=2) {
                    auto v=vaddq_f64(vmulq_f64(vmulq_f64(vld1q_f64(p+i*n+j),fi),vld1q_f64(phi+j)),vld1q_f64(q+i*n+j));
                    vst1q_f64(prediction_matrix+i*n+j,v);
                    if(ones) {total+=vgetq_lane_f64(v,0);total+=vgetq_lane_f64(v,1);}
                    else {total+=vgetq_lane_f64(v,0)*w[j];total+=vgetq_lane_f64(v,1)*w[j+1];}
                }
                for(;j<n;j++) {double v=p[i*n+j]*phi[i]*phi[j]+q[i*n+j];prediction_matrix[i*n+j]=v;total+=ones?v:v*w[j];}
                pw[i]=total;
            }
            variance=4.934802200544679+dot(&blas_n,const_cast<double*>(w),&stride,pw.data(),&stride);
            for(int i=0;i<n;i++) gain[i]=pw[i]/variance;
            log_variance=std::log(6.283185307179586*variance);
        }
        double prediction=0.;
        for(int i=0;i<n;i++) {s[i]*=phi[i];prediction+=ones?s[i]:w[i]*s[i];}
        double innovation=y[t]-level-prediction;
        for(int i=0;i<n;i++) s[i]+=gain[i]*innovation;
        if(!fixed) {
            fixed=true;
            auto vv=vdupq_n_f64(variance);
            for(int i=0;i<n;i++) {
                int j=0;auto pi=vdupq_n_f64(pw[i]);
                for(;j+1<n;j+=2) {
                    auto value=vsubq_f64(vld1q_f64(prediction_matrix+i*n+j),vdivq_f64(vmulq_f64(pi,vld1q_f64(pw.data()+j)),vv));
                    vst1q_f64(prediction_matrix+i*n+j,value);
                }
                for(;j<n;j++) {prediction_matrix[i*n+j]-=pw[i]*pw[j]/variance;}
            }
            for(int cell=0;cell<n*n;cell++) {if(prediction_matrix[cell]!=p[cell]) {fixed=false;break;}}
            std::swap(p,prediction_matrix);
        }
        loglik-=.5*(log_variance+innovation*innovation/variance);
    }
    if(p!=posterior.mutable_data())std::memcpy(posterior.mutable_data(),p,n*n*sizeof(double));
    }
    return py::make_tuple(loglik,state,posterior);
}
py::tuple filter(Array y, Array phi, Array w, Array q, double level, uintptr_t dot) {
    if(phi.size()==8)return filter_fixed<8>(y,phi,w,q,level,dot);
    if(phi.size()==33)return filter_fixed<33>(y,phi,w,q,level,dot);
    return filter_fixed<0>(y,phi,w,q,level,dot);
}
PYBIND11_MODULE(rough_overlay_native, m) {m.def("filter",&filter);}
