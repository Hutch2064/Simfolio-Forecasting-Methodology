#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
namespace py=pybind11;
using Array=py::array_t<double,py::array::c_style|py::array::forcecast>;
using Pot=void(*)(char*,int*,double*,int*,int*);
static Pot potrf=nullptr,potri=nullptr;
using Gesv=void(*)(int*,int*,double*,int*,int*,double*,int*,int*);
using Gemv=void(*)(char*,int*,int*,double*,double*,int*,double*,int*,double*,double*,int*);
using Dot=double(*)(int*,double*,int*,double*,int*);
static Gesv gesv=nullptr;static Gemv gemv=nullptr;static Dot dot=nullptr;
void configure(py::capsule rf,py::capsule ri,py::capsule sv,py::capsule mv,py::capsule dt){potrf=(Pot)PyCapsule_GetPointer(rf.ptr(),PyCapsule_GetName(rf.ptr()));potri=(Pot)PyCapsule_GetPointer(ri.ptr(),PyCapsule_GetName(ri.ptr()));gesv=(Gesv)PyCapsule_GetPointer(sv.ptr(),PyCapsule_GetName(sv.ptr()));gemv=(Gemv)PyCapsule_GetPointer(mv.ptr(),PyCapsule_GetName(mv.ptr()));dot=(Dot)PyCapsule_GetPointer(dt.ptr(),PyCapsule_GetName(dt.ptr()));}
py::tuple evaluate(Array z,Array s,double a,double b,Array initial,double nu,double constant,bool derivatives,py::object eta,double constant_eta,py::object initial_da,py::object initial_db,py::object initial_qe){
 int p=s.shape(0),n=z.shape(0);if(z.ndim()!=2||s.ndim()!=2||s.shape(1)!=p||z.shape(1)!=p||initial.ndim()!=2||initial.shape(0)!=p||initial.shape(1)!=p||!potrf||!potri)throw py::value_error("cDCC shape or uninitialized LAPACK");
 bool joint=!eta.is_none();Array ez;const double* ze=nullptr;if(joint){ez=eta.cast<Array>();if(ez.ndim()!=2||ez.shape(0)!=n||ez.shape(1)!=p)throw py::value_error("cDCC eta shape");ze=ez.data();}
 Array output({p,p}),gradient(joint?3:2);double* __restrict q=output.mutable_data();double* __restrict grad=gradient.mutable_data();for(int i=0;i<p*p;i++)q[i]=initial.data()[i];for(int i=0;i<(joint?3:2);i++)grad[i]=0.;if(joint)grad[2]=-n*constant_eta;
 const double* zs=z.data();const double* target=s.data();double loss=-n*constant,ga=0.,gb=0.,ge=joint?-n*constant_eta:0.;
 std::vector<double> da(p*p),db(p*p),qe(p*p),inv(p*p),sd(p),w(p),v(p),wa(p),wb(p),we(p);
 if(!initial_da.is_none()){Array x=initial_da.cast<Array>();if(x.size()!=p*p)throw py::value_error("initial derivative shape");da.assign(x.data(),x.data()+p*p);}
 if(!initial_db.is_none()){Array x=initial_db.cast<Array>();if(x.size()!=p*p)throw py::value_error("initial derivative shape");db.assign(x.data(),x.data()+p*p);}
 if(!initial_qe.is_none()){Array x=initial_qe.cast<Array>();if(x.size()!=p*p)throw py::value_error("initial derivative shape");qe.assign(x.data(),x.data()+p*p);}
 {py::gil_scoped_release release;
 double cached_logdet=0.,target_logdet=0.;
 bool rank_one=b==0. && a>0. && a<1.;
 std::vector<double> target_inv,previous_w(p),transformed(p);
 if(rank_one){
  target_inv.assign(target,target+p*p);int info=0;char triangle='L';
  potrf(&triangle,&p,target_inv.data(),&p,&info);if(info)throw std::runtime_error("nonpositive cDCC target");
  for(int i=0;i<p;i++)target_logdet+=2*std::log(target_inv[i*p+i]);
  potri(&triangle,&p,target_inv.data(),&p,&info);if(info)throw std::runtime_error("noninvertible cDCC target");
  for(int i=0;i<p;i++)for(int j=0;j<i;j++)target_inv[i*p+j]=target_inv[j*p+i];
 }
 for(int t=0;t<n;t++){
  const double* row=zs+t*p;double logdet=cached_logdet;
  if(rank_one && t>0){
   double quadratic=0.;
   for(int i=0;i<p;i++){double value=0.;for(int j=0;j<p;j++)value+=target_inv[i*p+j]*previous_w[j];transformed[i]=value;quadratic+=previous_w[i]*value;}
   double scale=1-a,denominator=1+a/scale*quadratic;
   logdet=target_logdet+p*std::log(scale)+std::log(denominator);
   double update=a/(scale*scale*denominator);
   for(int i=0;i<p;i++){sd[i]=std::sqrt(q[i*p+i]);logdet-=std::log(q[i*p+i]);for(int j=0;j<p;j++)inv[i*p+j]=target_inv[i*p+j]/scale-update*transformed[i]*transformed[j];}
  }else if(a!=0. || b!=0. || t<2){logdet=0.;for(int i=0;i<p*p;i++)inv[i]=q[i];int info=0;char triangle='L';potrf(&triangle,&p,inv.data(),&p,&info);if(info)throw std::runtime_error("nonpositive cDCC covariance");
  for(int i=0;i<p;i++){sd[i]=std::sqrt(q[i*p+i]);logdet+=2*std::log(inv[i*p+i])-std::log(q[i*p+i]);}
  potri(&triangle,&p,inv.data(),&p,&info);if(info)throw std::runtime_error("noninvertible cDCC covariance");
  for(int i=0;i<p;i++)for(int j=0;j<i;j++)inv[i*p+j]=inv[j*p+i];
  cached_logdet=logdet;}
  for(int i=0;i<p;i++){w[i]=sd[i]*row[i];previous_w[i]=w[i];}
  double maha=0.,marginal=0.;for(int i=0;i<p;i++){double value=0.;for(int j=0;j<p;j++)value+=inv[i*p+j]*w[j];v[i]=value;maha+=w[i]*value;if(nu>0)marginal+=std::log1p(row[i]*row[i]/(nu-2));else marginal+=row[i]*row[i];}
  double weight=nu>0?(nu+p)/(nu-2+maha):1.;if(nu>0)loss+=.5*logdet+.5*(nu+p)*std::log1p(maha/(nu-2))-.5*(nu+1)*marginal;else loss+=.5*(logdet+maha-marginal);
  if(derivatives){for(int i=0;i<p;i++){wa[i]=.5*row[i]/sd[i]*da[i*p+i];wb[i]=.5*row[i]/sd[i]*db[i*p+i];if(joint)we[i]=.5*row[i]/sd[i]*qe[i*p+i]+sd[i]*ze[t*p+i];}
   for(int i=0;i<p;i++)for(int j=0;j<=i;j++){double g=.5*inv[i*p+j]-.5*weight*v[i]*v[j];if(i==j)g+=-.5/q[i*p+i]+.5*weight*v[i]*row[i]/sd[i];double scale=i==j?1.:2.;ga+=scale*g*da[i*p+j];gb+=scale*g*db[i*p+j];if(joint)ge+=scale*g*qe[i*p+j];}}
  if(joint){double d=nu-2,tail=0.;for(int i=0;i<p;i++){double square=row[i]*row[i];tail+=square/(d*(d+square));ge+=weight*v[i]*sd[i]*ze[t*p+i]-(nu+1)*row[i]*ze[t*p+i]/(d+square);}ge+=d*(.5*std::log1p(maha/d)-.5*(nu+p)*maha/(d*(d+maha))-.5*marginal+.5*(nu+1)*tail);}
  for(int i=0;i<p;i++)for(int j=0;j<=i;j++){int ij=i*p+j,ji=j*p+i;double outer=w[i]*w[j],old=q[ij];if(derivatives){da[ij]=outer-target[ij]+a*(wa[i]*w[j]+w[i]*wa[j])+b*da[ij];db[ij]=old-target[ij]+a*(wb[i]*w[j]+w[i]*wb[j])+b*db[ij];}if(joint){qe[ij]=a*(we[i]*w[j]+w[i]*we[j])+b*qe[ij];}q[ij]=(1-a-b)*target[ij]+a*outer+b*old;q[ji]=q[ij];}
 }
 }
 grad[0]=ga;grad[1]=gb;if(joint)grad[2]=ge;
 return py::make_tuple(loss,gradient,output);
}
// State/derivative recursion is independent of likelihood factorizations.
// A cheap sequential pass supplies exact initial states to independent blocks.
Array starts(Array z,Array s,double a,double b,Array initial,int blocks,py::object eta){
 int p=s.shape(0),n=z.shape(0);if(blocks<1||blocks>n||s.size()!=p*p||z.shape(1)!=p||initial.size()!=p*p)throw py::value_error("block state shape");
 Array result({blocks,4,p,p});auto* out=result.mutable_data();const double* rows=z.data();const double* target=s.data();
 Array ez;const double* ze=nullptr;if(!eta.is_none()){ez=eta.cast<Array>();if(ez.size()!=z.size())throw py::value_error("block eta shape");ze=ez.data();}
 std::vector<double> q(initial.data(),initial.data()+p*p),da(p*p),db(p*p),qe(p*p),w(p),wa(p),wb(p),we(p);
 py::gil_scoped_release release;int block=0;
 for(int t=0;t<n;t++){
  if(block<blocks && t==block*n/blocks){for(int i=0;i<p*p;i++){out[(block*4)*p*p+i]=q[i];out[(block*4+1)*p*p+i]=da[i];out[(block*4+2)*p*p+i]=db[i];out[(block*4+3)*p*p+i]=qe[i];}block++;if(block==blocks)break;}
  const double* row=rows+t*p;
  for(int i=0;i<p;i++){double sd=std::sqrt(q[i*p+i]);w[i]=sd*row[i];wa[i]=.5*row[i]/sd*da[i*p+i];wb[i]=.5*row[i]/sd*db[i*p+i];if(ze)we[i]=.5*row[i]/sd*qe[i*p+i]+sd*ze[t*p+i];}
  for(int i=0;i<p;i++)for(int j=0;j<=i;j++){int ij=i*p+j,ji=j*p+i;double outer=w[i]*w[j],old=q[ij];da[ij]=outer-target[ij]+a*(wa[i]*w[j]+w[i]*wa[j])+b*da[ij];db[ij]=old-target[ij]+a*(wb[i]*w[j]+w[i]*wb[j])+b*db[ij];da[ji]=da[ij];db[ji]=db[ij];if(ze){qe[ij]=a*(we[i]*w[j]+w[i]*we[j])+b*qe[ij];qe[ji]=qe[ij];}q[ij]=(1-a-b)*target[ij]+a*outer+b*old;q[ji]=q[ij];}
 }
 return result;
}

// Preserve Numba's original upper-Cholesky, two general solves, BLAS dots,
// full-matrix gradient reductions, and scalar accumulation order. Reuse
// workspaces instead of allocating/copying temporary arrays per observation.
py::tuple gaussian_exact(Array z,Array s,double a,double b,Array initial){
 int p=s.shape(0),n=z.shape(0);if(z.ndim()!=2||s.ndim()!=2||s.shape(1)!=p||z.shape(1)!=p||initial.ndim()!=2||initial.shape(0)!=p||initial.shape(1)!=p||!gesv)throw py::value_error("cDCC shape or uninitialized LAPACK");
 Array output({p,p}),gradient(2);double* q=output.mutable_data();double* grad=gradient.mutable_data();grad[0]=grad[1]=0.;for(int i=0;i<p*p;i++)q[i]=initial.data()[i];
 const double* zs=z.data();const double* target=s.data();double loss=0.;
 std::vector<double> da(p*p),db(p*p),root(p*p),lu(p*p),inv(p*p),sd(p),w(p),v(p),wa(p),wb(p);std::vector<int> pivot(p);
 {py::gil_scoped_release release;
 for(int t=0;t<n;t++){
  const double* row=zs+t*p;for(int i=0;i<p*p;i++)root[i]=q[i];int info=0;char triangle='U';potrf(&triangle,&p,root.data(),&p,&info);if(info)throw std::runtime_error("nonpositive cDCC covariance");
  for(int i=0;i<p;i++)for(int j=i+1;j<p;j++)root[i*p+j]=0.;
  double logs=0.,diagonal=0.;for(int i=0;i<p;i++){sd[i]=std::sqrt(q[i*p+i]);w[i]=sd[i]*row[i];logs+=std::log(root[i*p+i]);diagonal+=std::log(q[i*p+i]);}
  for(int i=0;i<p;i++)for(int j=0;j<p;j++){lu[j*p+i]=root[i*p+j];inv[j*p+i]=i==j?1.:0.;}
  int rhs=p;gesv(&p,&rhs,lu.data(),&p,pivot.data(),inv.data(),&p,&info);if(info)throw std::runtime_error("noninvertible cDCC covariance");
  for(int i=0;i<p*p;i++)lu[i]=root[i];gesv(&p,&rhs,lu.data(),&p,pivot.data(),inv.data(),&p,&info);if(info)throw std::runtime_error("noninvertible cDCC covariance");
  char trans='N';int inc=1;double one=1.,zero=0.;gemv(&trans,&p,&p,&one,inv.data(),&p,w.data(),&inc,&zero,v.data(),&inc);
  double maha=dot(&p,w.data(),&inc,v.data(),&inc),marginal=dot(&p,const_cast<double*>(row),&inc,const_cast<double*>(row),&inc);double logdet=2*logs-diagonal;loss+=.5*(logdet+maha-marginal);
  for(int i=0;i<p;i++){wa[i]=.5*row[i]/sd[i]*da[i*p+i];wb[i]=.5*row[i]/sd[i]*db[i*p+i];}
  double ga=0.,gb=0.;for(int i=0;i<p;i++)for(int j=0;j<p;j++){double g=.5*inv[i+j*p]-.5*v[i]*v[j];if(i==j)g+=-.5/q[i*p+i]+.5*v[i]*row[i]/sd[i];ga+=g*da[i*p+j];gb+=g*db[i*p+j];}grad[0]+=ga;grad[1]+=gb;
  for(int i=0;i<p;i++)for(int j=0;j<p;j++){int ij=i*p+j;double outer=w[i]*w[j],old=q[ij];da[ij]=outer-target[ij]+a*(wa[i]*w[j]+w[i]*wa[j])+b*da[ij];db[ij]=old-target[ij]+a*(wb[i]*w[j]+w[i]*wb[j])+b*db[ij];q[ij]=(1-a-b)*target[ij]+a*outer+b*old;}
 }
 }
 return py::make_tuple(loss,gradient,output);
}
PYBIND11_MODULE(cdcc_lapack_native,m){m.def("configure",&configure);m.def("starts",&starts);m.def("gaussian_exact",&gaussian_exact);m.def("evaluate",&evaluate,py::arg("z"),py::arg("s"),py::arg("a"),py::arg("b"),py::arg("q0"),py::arg("nu"),py::arg("constant"),py::arg("derivatives")=true,py::arg("z_eta")=py::none(),py::arg("constant_eta")=0.,py::arg("initial_da")=py::none(),py::arg("initial_db")=py::none(),py::arg("initial_qe")=py::none());}
