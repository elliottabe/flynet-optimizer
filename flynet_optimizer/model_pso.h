#ifndef MODEL_PSO_H
#define MODEL_PSO_H

#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
#include <tuple>
#include <armadillo>
#include <stdlib.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <dirent.h>
#include <chrono>
#include <execution>
#include <math.h>

#include <unistd.h>
#define GetCurrentDir getcwd

#include "json.hpp"

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/call.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/clamp.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/thread.hpp>
#include <boost/container/vector.hpp>
#include <cstdio>
#include <ctime>

#include <vtkPolyData.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkCleanPolyData.h>
#include <vtkTriangleFilter.h>
#include <vtkButterflySubdivisionFilter.h>
#include <vtkLoopSubdivisionFilter.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkSphereSource.h>

namespace p = boost::python;
namespace np = boost::python::numpy;

#define PI 3.141592653589793

using namespace std;

using json = nlohmann::json;

class Cost {

	public:

		int N_files= 0;
		int N_components = 0;
		int N_segments = 0;
		int N_pixels = 0;
		int N_cam = 0;
		int comp_ind = 0;

		int calib_type;
		arma::Mat<double> world2cam;
		arma::Mat<double> w2c_scaling;
		arma::Col<int> window_size;
		arma::Mat<double> C_mat;
		arma::Mat<double> K_mat;
		arma::Mat<double> W2C_mat;
		arma::vec COM;
		arma::vec COM_uv;
		arma::vec uv_shift;
		double uv_cost;
		double sphere_radius;

		arma::Mat<double> state;
		arma::vec scale;
		arma::Mat<int> parent_child_ids;
		arma::Mat<int> body_or_wing;
		arma::Mat<double> M_scale;
		arma::Mat<double> M_state;
		vector<arma::Mat<double>> M_prev;

		double q0 = 0.0;
		double qx = 0.0;
		double qy = 0.0;
		double qz = 0.0;
		double q_norm = 0.0;
		double tx = 0.0;
		double ty = 0.0;
		double tz = 0.0;

		vector<arma::Mat<double>> mdl_pts;
		vector<arma::Mat<double>> mdl_pts_now;
		vector<arma::Mat<double>> mdl_pts_uv;
		vector<arma::Mat<int>> mdl_ids;

		arma::Mat<double> sphere_pts;
		arma::Mat<int> sphere_ids;
		arma::Mat<double> sphere_pts_now;
		arma::Mat<double> sphere_pts_uv;

		int pc_ind = 0;

		double u0 = 0.0;
		double v0 = 0.0;
		double u1 = 0.0;
		double v1 = 0.0;
		double u2 = 0.0;
		double v2 = 0.0;
		double u3 = 0.0;
		double v3 = 0.0;

		double A = 0.0;
		double A1 = 0.0;
		double A2 = 0.0;
		double A3 = 0.0;

		int min_u = 0;
		int min_v = 0;
		int max_u = 0;
		int max_v = 0;

		arma::vec u_tri;
		arma::vec v_tri;

		int N_u = 0;
		int N_v = 0;
		double N_u_2 = 0.0;
		double N_v_2 = 0.0;
		int N_pix = 0;
		int u_i = 0;
		int v_i = 0;
		int pix_i = 0;
		bool bounds_exceded;

		int N_vox_body;
		int N_vox_wing;
		int N_vox_diff;
		int N_vox_tot;
		int N_body_views;
		int N_wing_views;

		double body_img_count;
		double wing_img_count;
		double intersect_count;
		double diff_count;
		double union_count;
		double model_img_count;
		double overlap_diff_count;
		double overlap_intersect_count;
		double body_overlap_count;

		// Constraints:
		int c_type = 0;
		int p_ind  = 0;
		int c_ind  = 0;
		int jp_ind = 0;
		int jc_ind = 0;
		arma::vec s_P;
		arma::vec s_JP;
		arma::vec s_C;
		arma::vec s_JC;
		arma::Mat<double> Q_P;
		arma::vec q_Pn;
		arma::Mat<double> Q_C;
		arma::vec q_Cn;
		arma::Mat<double> Q_d;
		arma::vec q_d;
		double phi;
		double theta; 
		double eta;
		arma::vec t_P;
		arma::Mat<double> M_P;
		arma::vec t_JP;
		arma::Mat<double> M_Pn;
		arma::vec t_Pn;
		arma::vec t_C;
		arma::Mat<double> M_C;
		arma::vec t_JC;
		arma::Mat<double> M_Cn;
		arma::vec t_Cn;
		arma::vec t_d;
		double dist;

		boost::dynamic_bitset<> overlap_bitset_body;
		boost::dynamic_bitset<> overlap_bitset_wing;
		boost::dynamic_bitset<> temp_image;
		boost::container::vector<boost::dynamic_bitset<>> body_image;
		boost::container::vector<boost::dynamic_bitset<>> wing_image;
		boost::container::vector<boost::dynamic_bitset<>> mdl_image;
		boost::container::vector<boost::dynamic_bitset<>> uv_shift_image;

		arma::Mat<double> cost_state;

		Cost();

		void load_stl_files(vector<string> file_names);
		void set_scale(arma::vec scale_in, arma::Mat<int> pc_ids_in, arma::Mat<int> body_or_wing_in);
		vtkSmartPointer<vtkPolyData> load_stl(const char* input_file);
		arma::Mat<int> get_ids(vtkSmartPointer<vtkPolyData> polydata_in);
		arma::Mat<double> get_pts(vtkSmartPointer<vtkPolyData> polydata_in);
		void set_calibration(int calib_type_in, arma::Mat<double> c_params_in, arma::Col<int> window_size_in, int N_cam_in);
		void set_COM(arma::vec com_in);
		void set_bin_imgs(boost::container::vector<boost::dynamic_bitset<>> bin_img_body_in,boost::container::vector<boost::dynamic_bitset<>> bin_img_wing_in);
		void set_state(arma::Mat<double> state_in);
		void set_comp_ind(int comp_ind_in);
		arma::Mat<double> compute_cost();

};

class MDL_PSO
{
	
	public:

		// model location:
		string model_loc = "/home/flyami/Documents/FlyNet4/models/drosophila";
		// variables
		double w;
		double c1;
		double c2;
		double c3;
		int N_dim=0;
		int N_particles=0;
		int N_state=0;
		int N_iter=0;
		int N_constraints=0;
		int N_joints=0;

		int N_cam;
		int Calib_type;
		arma::Mat<double> c_params;
		arma::Col<int> window_size;
		arma::vec uv_shift;
		arma::vec scale;
		arma::Mat<double> COM;
		int N_batch=0;
		int N_pixels=0;
		int N_components=0;
		int N_segments=0;
		vector<string> file_names;

		boost::container::vector<boost::dynamic_bitset<>> bin_img_body;
		boost::container::vector<boost::dynamic_bitset<>> bin_img_wing;
		boost::container::vector<boost::dynamic_bitset<>> temp_img_body;
		boost::container::vector<boost::dynamic_bitset<>> temp_img_wing;

		arma::Mat<double> constraint_ids; 
		arma::Mat<double> joint_locs; 
		arma::Mat<int> constraint_types;
		arma::Mat<double> constraint_bounds;
		arma::Mat<int> constraint_p_comp;
		double constraint_tol = 1e-3;

		arma::Mat<double> init_state;
		arma::Mat<double> state_const;
		arma::Mat<int> state_conn;
		arma::Mat<int> state_comp;
		arma::Mat<int> parent_child_ids;
		arma::Mat<int> body_or_wing;
		arma::vec state_dev;
		arma::Mat<double> state_bounds;
		arma::Mat<double> state_best;

		arma::Mat<double> positions;
		arma::Mat<double> pb_positions;
		arma::Mat<double> velocities;
		arma::Mat<double> gb_positions;
		arma::Mat<double> personal_best;
		arma::Mat<double> global_best;
		arma::Mat<double> cost_mat;
		arma::Row<int> 	  comp_indices;
		arma::vec rand_ids;
		arma::vec pb_update;
		arma::Mat<double> rand_velocities;
		vector<arma::Mat<double>> cost_vec;
		arma::Mat<double> state_j;

		vector<Cost> cost_threads;

		MDL_PSO();

		void set_PSO_param(double w_in,double c1_in,double c2_in,double c3_in,int N_dim_in,int N_particles_in,int N_iter_in);
		void load_model_json(string model_loc, string model_file_name);
		void set_scale(np::ndarray scale_in, int N_comp_in);
		void set_COM(np::ndarray COM_in);
		void set_calibration(int calib_type_in, np::ndarray c_params_in, np::ndarray window_size_in, int N_cam_in) ;
		void set_batch_size(int N_batch_in);
		void set_bin_imgs(np::ndarray body_mask,np::ndarray wing_mask,int b_ind,int n_ind,int n_body,int n_wing);
		void set_particle_start(np::ndarray init_state_in);
		arma::vec quat_SLERP(arma::vec qA,arma::vec qB,double t);
		arma::vec quat_diff(arma::vec qA, arma::vec qB);
		arma::vec quat_multiply(arma::vec qA, arma::vec qB);
		arma::vec transform_matrix(arma::vec sA, arma::vec tB, int trans);
		bool check_constraints(arma::vec state_in);
		arma::vec set_constraints(arma::vec s_P, arma::vec s_JP, arma::vec s_C, arma::vec s_JC, arma::Mat<double> c_bounds, arma::Row<int> c_types);
		arma::vec quaternion_check(arma::vec q_in);
		void position_update(int N_neighbors,int iter_ind,int iter_max);
		arma::Mat<double> compute_state(arma::vec x);
		void setup_threads(string model_loc);
		np::ndarray PSO_fit();
		np::ndarray return_model_img();
};
#endif