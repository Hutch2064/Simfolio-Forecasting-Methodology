#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
namespace py=pybind11;
using Array=py::array_t<double,py::array::c_style>;

// The input normals and output scores use (day, path, asset). Each path's
// Q is carried across numerical memory blocks. No statistical work is omitted.
void scores(Array normals,Array target,Array states,double a,double b) {
    if(normals.ndim()!=3 || target.ndim()!=2 || states.ndim()!=3)
        throw py::value_error("cDCC array dimensions");
    auto days=normals.shape(0),sims=normals.shape(1),p=normals.shape(2);
    if(target.shape(0)!=p || target.shape(1)!=p || states.shape(0)!=sims ||
       states.shape(1)!=p || states.shape(2)!=p || a<0. || b<0. || a+b>=1.)
        throw py::value_error("cDCC shape or stationarity");
    double* z=normals.mutable_data();double* qs=states.mutable_data();
    const double* s=target.data();double c=1.-a-b;
    std::vector<double> root(p*p),w(p),sd(p);
    py::gil_scoped_release release;
    for(py::ssize_t t=0;t<days;t++)for(py::ssize_t k=0;k<sims;k++) {
        double* q=qs+k*p*p;double* row=z+(t*sims+k)*p;
        for(py::ssize_t i=0;i<p;i++) {
            sd[i]=std::sqrt(q[i*p+i]);
            for(py::ssize_t j=0;j<=i;j++) {
                double v=q[i*p+j];
                for(py::ssize_t h=0;h<j;h++)v-=root[i*p+h]*root[j*p+h];
                if(i==j) {
                    if(!(v>0.) || !std::isfinite(v))throw std::runtime_error("cDCC lost positive definiteness");
                    root[i*p+j]=std::sqrt(v);
                } else root[i*p+j]=v/root[j*p+j];
            }
            double v=0.;for(py::ssize_t j=0;j<=i;j++)v+=root[i*p+j]*row[j];
            w[i]=v;
        }
        for(py::ssize_t i=0;i<p;i++)row[i]=w[i]/sd[i];
        for(py::ssize_t i=0;i<p;i++)for(py::ssize_t j=0;j<=i;j++) {
            double v=c*s[i*p+j]+a*w[i]*w[j]+b*q[i*p+j];
            q[i*p+j]=v;q[j*p+i]=v;
        }
    }
}
PYBIND11_MODULE(corrected_dcc_native,m){m.def("scores",&scores);}
