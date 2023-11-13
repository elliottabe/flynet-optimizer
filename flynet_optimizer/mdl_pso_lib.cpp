#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/call.hpp>
#include "model_pso.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(_mdl_pso_lib)
{
	Py_Initialize();
	np::initialize();
	p::class_<MDL_PSO>("MDL_PSO")
		.def("set_PSO_param",&MDL_PSO::set_PSO_param)
		.def("set_scale",&MDL_PSO::set_scale)
		.def("set_COM",&MDL_PSO::set_COM)
		.def("set_calibration",&MDL_PSO::set_calibration)
		.def("set_batch_size",&MDL_PSO::set_batch_size)
		.def("set_bin_imgs",&MDL_PSO::set_bin_imgs)
		.def("set_particle_start",&MDL_PSO::set_particle_start)
		.def("load_model_json",&MDL_PSO::load_model_json)
		.def("setup_threads",&MDL_PSO::setup_threads)
		.def("PSO_fit",&MDL_PSO::PSO_fit)
		.def("return_model_img",&MDL_PSO::return_model_img);
}
