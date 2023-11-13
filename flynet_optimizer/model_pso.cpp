#include "model_pso.h"

string GetCurrentWorkingDir( void ) {
	char buff[FILENAME_MAX];
	GetCurrentDir( buff, FILENAME_MAX );
	string current_working_dir(buff);
	return current_working_dir;
}

Cost::Cost() {
}

void Cost::load_stl_files(vector<string> file_names) {
	N_files = file_names.size();
	mdl_pts.clear();
	mdl_ids.clear();
	mdl_pts_now.clear();
	mdl_pts_uv.clear();
	state.zeros(7,N_files);
	arma::Mat<double> uv_temp;
	for (int i=0; i<N_files; i++) {
		vtkSmartPointer<vtkPolyData> poly = Cost::load_stl(file_names[i].c_str());
		arma::Mat<double> poly_pts = Cost::get_pts(poly);
		arma::Mat<int> poly_ids = Cost::get_ids(poly);
		mdl_pts.push_back(poly_pts);
		mdl_ids.push_back(poly_ids);
		mdl_pts_now.push_back(poly_pts);
		uv_temp.zeros(N_cam*2,poly_pts.n_cols);
		mdl_pts_uv.push_back(uv_temp);
	}
}

void Cost::set_scale(arma::vec scale_in, arma::Mat<int> pc_ids_in, arma::Mat<int> body_or_wing_in) {
	N_components = scale_in.n_rows;
	N_segments = pc_ids_in.n_cols;
	scale = scale_in;
	parent_child_ids = pc_ids_in; // size = (N_segments,N_components) (non-components are -1)
	body_or_wing = body_or_wing_in;
	M_scale.eye(4,4);
	M_state.eye(4,4);
	u_tri.zeros(3);
	v_tri.zeros(3);
	cost_state.zeros(N_components,1);
	// set mdl image
	mdl_image.clear();
	for (int i=0; i<(N_components*N_cam); i++) {
		mdl_image.push_back(temp_image);
	}
}

vtkSmartPointer<vtkPolyData> Cost::load_stl(const char* input_file) {
	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
	reader->SetFileName(input_file);
	reader->Update();
	vtkSmartPointer<vtkPolyData> polydataCopy = vtkSmartPointer<vtkPolyData>::New();
	polydataCopy->DeepCopy(reader->GetOutput());
	return polydataCopy;
}

arma::Mat<int> Cost::get_ids(vtkSmartPointer<vtkPolyData> polydata_in) {
	int N_cells = polydata_in->GetNumberOfCells();
	arma::Mat<int> tri_ids;
	tri_ids.zeros(3,N_cells);
	for(vtkIdType i = 0; i < N_cells; i++) {
		tri_ids(0,i) = polydata_in->GetCell(i)->GetPointIds()->GetId(0);
		tri_ids(1,i) = polydata_in->GetCell(i)->GetPointIds()->GetId(1);
		tri_ids(2,i) = polydata_in->GetCell(i)->GetPointIds()->GetId(2);
	}
	return tri_ids;
}

arma::Mat<double> Cost::get_pts(vtkSmartPointer<vtkPolyData> polydata_in) {
	int N_pts = polydata_in->GetNumberOfPoints();
	arma::Mat<double> pts_out;
	pts_out.ones(4,N_pts);
	for(vtkIdType i = 0; i < N_pts; i++) {
		double p[3];
		polydata_in->GetPoint(i,p);
		pts_out(0,i) = p[0];
		pts_out(1,i) = p[1];
		pts_out(2,i) = p[2];
		pts_out(3,i) = 1.0;
	}
	return pts_out;
}

void Cost::set_calibration(int calib_type_in, arma::Mat<double> c_params_in, arma::Col<int> window_size_in, int N_cam_in) {
	// N_cam
	calib_type = calib_type_in;
	world2cam.reset();
	w2c_scaling.reset();
	window_size.reset();
	N_cam = N_cam_in;
	window_size.zeros(2*N_cam);
	world2cam.zeros(N_cam*2,4);
	w2c_scaling.zeros(N_cam*2,4);
	C_mat.zeros(3,4);
	K_mat.zeros(4,4);
	W2C_mat.zeros(3,4);
	N_pixels = 0;
	if (calib_type==0) {
		for (int n=0; n<N_cam; n++) {
			// window size
			window_size(n*2) = window_size_in(n*2);
			window_size(n*2+1) = window_size_in(n*2+1);
			N_pixels = window_size(n*2)*window_size(n*2+1);
			// projection
			world2cam(n*2,0) = c_params_in(0,n);
			world2cam(n*2,1) = c_params_in(1,n);
			world2cam(n*2,2) = c_params_in(2,n);
			world2cam(n*2,3) = c_params_in(3,n);
			world2cam(n*2+1,0) = c_params_in(4,n);
			world2cam(n*2+1,1) = c_params_in(5,n);
			world2cam(n*2+1,2) = c_params_in(6,n);
			world2cam(n*2+1,3) = c_params_in(7,n);
			// scaling
			w2c_scaling(n*2,0) = c_params_in(8,n);
			w2c_scaling(n*2,1) = c_params_in(9,n);
			w2c_scaling(n*2,2) = c_params_in(10,n);
			w2c_scaling(n*2,3) = 1.0;
			w2c_scaling(n*2+1,0) = c_params_in(8,n);
			w2c_scaling(n*2+1,1) = c_params_in(9,n);
			w2c_scaling(n*2+1,2) = c_params_in(10,n);
			w2c_scaling(n*2+1,3) = 1.0;
		}
		temp_image.resize(N_pixels);
		overlap_bitset_body.resize(N_pixels);
		overlap_bitset_wing.resize(N_pixels);
	}
	else if (calib_type==1) {
		for (int n=0; n<N_cam; n++) {
			// window size
			window_size(n*2) = window_size_in(n*2);
			window_size(n*2+1) = window_size_in(n*2+1);
			N_pixels = window_size(n*2)*window_size(n*2+1);
			C_mat(0,0) = c_params_in(0,n);
			C_mat(1,1) = c_params_in(1,n);
			C_mat(0,1) = c_params_in(2,n);
			C_mat(2,3) = 1.0;
			q0 = c_params_in(3,n);
			qx = c_params_in(4,n);
			qy = c_params_in(5,n);
			qz = c_params_in(6,n);
			tx = c_params_in(7,n);
			ty = c_params_in(8,n);
			tz = c_params_in(9,n);
			K_mat(0,0) = 2.0*pow(q0,2)-1.0+2.0*pow(qx,2);
			K_mat(0,1) = 2.0*qx*qy+2.0*q0*qz;
			K_mat(0,2) = 2.0*qx*qz-2.0*q0*qy;
			K_mat(0,3) = tx;
			K_mat(1,0) = 2.0*qx*qy-2.0*q0*qz;
			K_mat(1,1) = 2.0*pow(q0,2)-1.0+2.0*pow(qy,2);
			K_mat(1,2) = 2.0*qy*qz+2.0*q0*qx;
			K_mat(1,3) = ty;
			K_mat(2,0) = 2.0*qx*qz+2.0*q0*qy;
			K_mat(2,1) = 2.0*qy*qz-2.0*q0*qx;
			K_mat(2,2) = 2.0*pow(q0,2)-1.0+2.0*pow(qz,2);
			K_mat(2,3) = tz;
			K_mat(3,0) = 0.0;
			K_mat(3,1) = 0.0;
			K_mat(3,2) = 0.0;
			K_mat(3,3) = 1.0;
			W2C_mat = C_mat*K_mat;
			W2C_mat(0,3) = W2C_mat(0,3)-c_params_in(11,n)/2.0+c_params_in(13,n)/2.0+c_params_in(14,n);
			W2C_mat(1,3) = W2C_mat(1,3)-c_params_in(10,n)/2.0+c_params_in(12,n)/2.0+c_params_in(15,n);
			// projection
			world2cam(n*2,0) = W2C_mat(0,0);
			world2cam(n*2,1) = W2C_mat(0,1);
			world2cam(n*2,2) = W2C_mat(0,2);
			world2cam(n*2,3) = W2C_mat(0,3);
			world2cam(n*2+1,0) = W2C_mat(1,0);
			world2cam(n*2+1,1) = W2C_mat(1,1);
			world2cam(n*2+1,2) = W2C_mat(1,2);
			world2cam(n*2+1,3) = W2C_mat(1,3);
		}
		temp_image.resize(N_pixels);
		overlap_bitset_body.resize(N_pixels);
		overlap_bitset_wing.resize(N_pixels);
	}
}

void Cost::set_COM(arma::vec com_in) {
	COM.reset();
	COM.zeros(4);
	COM(0) = com_in(0);
	COM(1) = com_in(1);
	COM(2) = com_in(2);
	COM(3) = 1.0;
	if (calib_type==0) {
		COM_uv = ((world2cam*COM)/(w2c_scaling*COM));
	}
	else if (calib_type==1) {
		COM_uv = world2cam*COM;
	}
}

void Cost::set_bin_imgs(boost::container::vector<boost::dynamic_bitset<>> bin_img_body_in,boost::container::vector<boost::dynamic_bitset<>> bin_img_wing_in) {
	body_image.clear();
	wing_image.clear();
	body_image = bin_img_body_in;
	wing_image = bin_img_wing_in;
}

void Cost::set_state(arma::Mat<double> state_in) {
	state = state_in;
}

void Cost::set_comp_ind(int comp_ind_in) {
	comp_ind = comp_ind_in;
}

arma::Mat<double> Cost::compute_cost() {
	bounds_exceded = false;
	int i = comp_ind;
	M_scale(0,0) = scale(i);
	M_scale(1,1) = scale(i);
	M_scale(2,2) = scale(i);
	M_scale(3,3) = 1.0;
	M_prev.clear();
	for (int n=0; n<N_cam; n++) {
		mdl_image.at(i*N_cam+n).reset();
	}
	N_vox_body = 0;
	N_vox_wing = 0;
	N_vox_diff = 0;
	N_vox_tot = 0;
	for (int j=0; j<N_segments; j++) {
		pc_ind = parent_child_ids(i,j);
		if (pc_ind >= 0) {
			if (j==0) {
				M_state(0,0) = 2.0*pow(state(0,pc_ind),2)-1.0+2.0*pow(state(1,pc_ind),2);
				M_state(0,1) = 2.0*state(1,pc_ind)*state(2,pc_ind)-2.0*state(0,pc_ind)*state(3,pc_ind);
				M_state(0,2) = 2.0*state(1,pc_ind)*state(3,pc_ind)+2.0*state(0,pc_ind)*state(2,pc_ind);
				M_state(0,3) = state(4,pc_ind)+COM(0);
				M_state(1,0) = 2.0*state(1,pc_ind)*state(2,pc_ind)+2.0*state(0,pc_ind)*state(3,pc_ind);
				M_state(1,1) = 2.0*pow(state(0,pc_ind),2)-1.0+2.0*pow(state(2,pc_ind),2);
				M_state(1,2) = 2.0*state(2,pc_ind)*state(3,pc_ind)-2.0*state(0,pc_ind)*state(1,pc_ind);
				M_state(1,3) = state(5,pc_ind)+COM(1);
				M_state(2,0) = 2.0*state(1,pc_ind)*state(3,pc_ind)-2.0*state(0,pc_ind)*state(2,pc_ind);
				M_state(2,1) = 2.0*state(2,pc_ind)*state(3,pc_ind)+2.0*state(0,pc_ind)*state(1,pc_ind);
				M_state(2,2) = 2.0*pow(state(0,pc_ind),2)-1.0+2.0*pow(state(3,pc_ind),2);
				M_state(2,3) = state(6,pc_ind)+COM(2);
				M_state(3,0) = 0.0;
				M_state(3,1) = 0.0;
				M_state(3,2) = 0.0;
				M_state(3,3) = 1.0;
				mdl_pts_now[pc_ind] = M_state*(M_scale*mdl_pts[pc_ind]);
				if (calib_type==0) {
					mdl_pts_uv[pc_ind] = (world2cam*mdl_pts_now[pc_ind])/(w2c_scaling*mdl_pts_now[pc_ind]);
				}
				else if (calib_type==1) {
					mdl_pts_uv[pc_ind] = world2cam*mdl_pts_now[pc_ind];
				}
				M_prev.push_back(M_state);
			}
			else  {
				M_state(0,0) = 2.0*pow(state(0,pc_ind),2)-1.0+2.0*pow(state(1,pc_ind),2);
				M_state(0,1) = 2.0*state(1,pc_ind)*state(2,pc_ind)-2.0*state(0,pc_ind)*state(3,pc_ind);
				M_state(0,2) = 2.0*state(1,pc_ind)*state(3,pc_ind)+2.0*state(0,pc_ind)*state(2,pc_ind);
				M_state(0,3) = state(4,pc_ind);
				M_state(1,0) = 2.0*state(1,pc_ind)*state(2,pc_ind)+2.0*state(0,pc_ind)*state(3,pc_ind);
				M_state(1,1) = 2.0*pow(state(0,pc_ind),2)-1.0+2.0*pow(state(2,pc_ind),2);
				M_state(1,2) = 2.0*state(2,pc_ind)*state(3,pc_ind)-2.0*state(0,pc_ind)*state(1,pc_ind);
				M_state(1,3) = state(5,pc_ind);
				M_state(2,0) = 2.0*state(1,pc_ind)*state(3,pc_ind)-2.0*state(0,pc_ind)*state(2,pc_ind);
				M_state(2,1) = 2.0*state(2,pc_ind)*state(3,pc_ind)+2.0*state(0,pc_ind)*state(1,pc_ind);
				M_state(2,2) = 2.0*pow(state(0,pc_ind),2)-1.0+2.0*pow(state(3,pc_ind),2);
				M_state(2,3) = state(6,pc_ind);
				M_state(3,0) = 0.0;
				M_state(3,1) = 0.0;
				M_state(3,2) = 0.0;
				M_state(3,3) = 1.0;
				mdl_pts_now[pc_ind] = M_state*(M_scale*mdl_pts[pc_ind]);
				for (int k=j-1; k>=0; k--) {
					mdl_pts_now[pc_ind] = M_prev[k]*mdl_pts_now[pc_ind];
				}
				if (calib_type==0) {
					mdl_pts_uv[pc_ind] = (world2cam*mdl_pts_now[pc_ind])/(w2c_scaling*mdl_pts_now[pc_ind]);
				}
				else if (calib_type==1) {
					mdl_pts_uv[pc_ind] = world2cam*mdl_pts_now[pc_ind];
				}
				M_prev.push_back(M_state);
			}
			N_pix = 0;
			for (int k=0; k<mdl_ids[pc_ind].n_cols; k++) {
				N_body_views = 0;
				N_wing_views = 0;
				for (int n=0; n<N_cam; n++) {
					N_u = window_size(n*2+1);
					N_v = window_size(n*2);
					N_u_2 = N_u/2.0-COM_uv(n*2);
					N_v_2 = N_v/2.0+COM_uv(n*2+1);
					N_pix = N_u*N_v;
					u1 = N_u_2+mdl_pts_uv[pc_ind](n*2,mdl_ids[pc_ind](0,k));
					v1 = N_v_2-mdl_pts_uv[pc_ind](n*2+1,mdl_ids[pc_ind](0,k));
					u2 = N_u_2+mdl_pts_uv[pc_ind](n*2,mdl_ids[pc_ind](1,k));
					v2 = N_v_2-mdl_pts_uv[pc_ind](n*2+1,mdl_ids[pc_ind](1,k));
					u3 = N_u_2+mdl_pts_uv[pc_ind](n*2,mdl_ids[pc_ind](2,k));
					v3 = N_v_2-mdl_pts_uv[pc_ind](n*2+1,mdl_ids[pc_ind](2,k));
					u_tri(0) = u1;
					u_tri(1) = u2;
					u_tri(2) = u3;
					v_tri(0) = v1;
					v_tri(1) = v2;
					v_tri(2) = v3;
					A = abs((u1*(v2-v3)+u2*(v3-v1)+u3*(v1-v2))/2.0);
					min_u = (int) clamp(floor(u_tri.min()),0.0,1.0*window_size(n*2+1)-1.0);
					if (min_u==0) {
						bounds_exceded = true;
					}
					min_v = (int) clamp(floor(v_tri.min()),0.0,1.0*window_size(n*2)-1.0);
					if (min_v==0) {
						bounds_exceded = true;
					}
					max_u = (int) clamp(ceil(u_tri.max()),0.0,1.0*window_size(n*2+1)-1.0);
					if (max_u==(window_size(n*2+1)-1)) {
						bounds_exceded = true;
					}
					max_v = (int) clamp(ceil(v_tri.max()),0.0,1.0*window_size(n*2)-1.0);
					if (max_v==(window_size(n*2)-1)) {
						bounds_exceded = true;
					}
					if (bounds_exceded==false) {
						for (int l=min_v; l<max_v; l++) {
							for (int m=min_u; m<max_u; m++) {
								pix_i = N_u*l+m;
								u0 = m*1.0+0.5;
								v0 = l*1.0+0.5;
								A1 = abs((u0*(v2-v3)+u2*(v3-v0)+u3*(v0-v2))/2.0); 
								A2 = abs((u1*(v0-v3)+u0*(v3-v1)+u3*(v1-v0))/2.0); 
								A3 = abs((u1*(v2-v0)+u2*(v0-v1)+u0*(v1-v2))/2.0); 
								if (abs(A-A1-A2-A3)<1.0e-6) {
									mdl_image.at(i*N_cam+n).set(pix_i);
								}
							}
						}
						// Check how many voxels are inside the body & wing:
						u0 = arma::mean(u_tri);
						v0 = arma::mean(v_tri);
						pix_i = N_u*((int) v0)+((int) u0);
						if (body_image.at(n).test(pix_i)) {
							N_body_views++;
							N_wing_views++;
						}
						else if (wing_image.at(n).test(pix_i)) {
							N_wing_views++;
						}
					}
				}
				if (body_or_wing(i,0)==0) {
					if (N_body_views==N_cam) {
						N_vox_body++;
					}
					else {
						N_vox_diff++;
					}
					N_vox_tot++;
				}
				else {
					if (N_wing_views==N_cam) {
						if (N_body_views==N_cam) {
							N_vox_diff++;
						}
						else {
							N_vox_wing++;
						}
					}
					else {
						N_vox_diff++;
					}
					N_vox_tot++;
				}
			}
		}
	}
	// Compute cost function:
	cost_state.fill(1.0e9);
	if (bounds_exceded==false) {
		if (body_or_wing(i,0)==0) {
			cost_state(i,0) = 0.0;
			for (int n=0; n<N_cam; n++) {
				overlap_bitset_body.reset();
				for (int j=0; j<N_components; j++) {
					if (body_or_wing(j,0)==0) {
						if (j==0) {
							overlap_bitset_body = mdl_image.at(j*N_cam+n);
						}
						else {
							overlap_bitset_body = overlap_bitset_body|mdl_image.at(j*N_cam+n);
						}
					}
				}
				diff_count = (body_image.at(n)^overlap_bitset_body).count()*1.0;
				//intersect_count = (body_image.at(n)&mdl_image.at(i*N_cam+n)).count()*1.0;
				cost_state(i,0) += (diff_count+N_vox_diff)/(N_cam*1.0);
				//cost_state(i,0) += (diff_count+N_vox_diff)/(overlap_bitset_body.count()*N_cam*1.0);
				//cost_state(i,0) += (diff_count)/((overlap_bitset_body.count()+intersect_count)*N_cam*1.0);
				//cost_state(i,0) += (diff_count)/(N_cam*1.0);
			}
		}
		else {
			cost_state(i,0) = 0.0;
			for (int n=0; n<N_cam; n++) {
				overlap_bitset_wing.reset();
				for (int j=0; j<N_components; j++) {
					if (body_or_wing(j,0)==1) {
						if (j==3) {
							overlap_bitset_wing = mdl_image.at(j*N_cam+n);
						}
						else {
							overlap_bitset_wing = overlap_bitset_wing|mdl_image.at(j*N_cam+n);
						}
					}
				}
				diff_count = (wing_image.at(n)^overlap_bitset_wing).count()*1.0;
				//intersect_count = (wing_image.at(n)&mdl_image.at(i*N_cam+n)).count()*1.0;
				//wing_img_count = wing_image.at(n).count()*1.0;
				body_overlap_count = (body_image.at(n)&overlap_bitset_wing).count()*1.0;
				cost_state(i,0) += (abs(diff_count-body_overlap_count)+N_vox_diff)/(N_cam*1.0);
				//cost_state(i,0) += abs(diff_count-body_overlap_count)/(N_cam*1.0);
			}
		}
	}
	return cost_state;
}

MDL_PSO::MDL_PSO() {
}

void MDL_PSO::set_PSO_param(double w_in,double c1_in,double c2_in,double c3_in,int N_dim_in,int N_particles_in,int N_iter_in) {
	w = w_in;
	c1 = c1_in;
	c2 = c2_in;
	c3 = c3_in;
	N_state = N_dim_in;
	N_particles = N_particles_in;
	N_iter = N_iter_in;

	positions.zeros(N_state,N_particles*N_batch);
	pb_positions.zeros(N_state,N_particles*N_batch);
	velocities.zeros(N_state,N_particles*N_batch);
	gb_positions.zeros(N_state,N_particles*N_batch);
	personal_best.zeros(N_components,N_particles*N_batch);
	global_best.zeros(N_components,N_batch);
	cost_mat.zeros(N_components,N_particles*N_batch);
	comp_indices.zeros(N_particles*N_batch);
	rand_ids.zeros(N_particles*N_batch);
	pb_update.zeros(N_state);
	rand_velocities.zeros(N_state,N_particles);
	cost_vec.resize(N_particles*N_batch);
}

void MDL_PSO::load_model_json(string model_loc, string model_file_name) {
	// Change directory
	chdir(model_loc.c_str());
	// Load JSON file:
	ifstream in_file(model_file_name.c_str());
	json model_file;
	in_file >> model_file;
	// Set parameters:
	N_state = model_file["N_state"];
	N_segments = model_file["N_segments"];
	N_components = model_file["N_components"];
	state_const.zeros(7,N_segments);
	state_conn.zeros(7,N_segments);
	for (int i=0; i<N_segments; i++) {
		for (int j=0; j<7; j++) {
			state_const(j,i) = model_file["state_constants"][j][i];
			state_conn(j,i) = model_file["state_connectivity"][j][i];
		}
	}
	state_dev.zeros(N_state);
	state_bounds.zeros(N_state,2);
	for (int i=0; i<N_state; i++) {
		state_dev(i) = model_file["std_dev"][i];
		state_bounds(i,0) = model_file["bounds"][i][0];
		state_bounds(i,1) = model_file["bounds"][i][1];
	}
	state_comp.zeros(N_components,2);
	parent_child_ids.zeros(N_components,N_segments);
	body_or_wing.zeros(N_components,N_segments);
	for (int i=0; i<N_components; i++) {
		state_comp(i,0) = model_file["state_computation"][i][0];
		state_comp(i,1) = model_file["state_computation"][i][1];
		for (int j=0; j<N_segments; j++) {
			parent_child_ids(i,j) = model_file["parent_child_inds"][i][j];
			body_or_wing(i,j) = model_file["body_or_wing"][i][j];
		}
	}
	file_names.clear();
	for (int i=0; i<N_segments; i++) {
		string temp_string = model_file["stl_file_names"][i];
		file_names.push_back(temp_string);
		//cout << temp_string << endl;
	}
	N_joints = model_file["N_joints"];
	joint_locs.zeros(7,N_joints);
	for (int i=0; i<N_joints; i++) {
		for (int j=0; j<7; j++) {
			joint_locs(j,i) = model_file["joint_locs"][i][j];
		}
	}
	N_constraints = model_file["N_constraints"];
	constraint_ids.zeros(N_constraints,2);
	constraint_types.zeros(N_constraints,5);
	constraint_tol = model_file["constraint_tol"];
	constraint_bounds.zeros(N_constraints*7,2);
	for (int i=0; i<N_constraints; i++) {
		constraint_ids(i,0) = model_file["constraint_ids"][i][0];
		constraint_ids(i,1) = model_file["constraint_ids"][i][1];
		for (int j=0; j<7; j++) {
			joint_locs(j,i) = model_file["joint_locs"][i][j];
			constraint_bounds(i*7+j,0) = model_file["constraint_bounds"][i*7+j][0];
			constraint_bounds(i*7+j,1) = model_file["constraint_bounds"][i*7+j][1];
		}
		constraint_types(i,0) = model_file["constraint_types"][i][0];
		constraint_types(i,1) = model_file["constraint_types"][i][1];
		constraint_types(i,2) = model_file["constraint_types"][i][2];
		constraint_types(i,3) = model_file["constraint_types"][i][3];
		constraint_types(i,4) = model_file["constraint_types"][i][4];
	}
}

void MDL_PSO::set_scale(np::ndarray scale_in, int N_comp_in) {
	scale.reset();
	N_components = N_comp_in;
	scale.zeros(N_components);
	for (int i=0; i<N_components; i++) {
		scale(i) = p::extract<double>(scale_in[i]);
	}
}

void MDL_PSO::set_COM(np::ndarray COM_in) {
	COM.zeros(4,N_batch);
	for (int i=0; i<N_batch; i++) {
		COM(0,i) = p::extract<double>(COM_in[0][i]);
		COM(1,i) = p::extract<double>(COM_in[1][i]);
		COM(2,i) = p::extract<double>(COM_in[2][i]);
		COM(3,i) = 1.0;
	}
}

void MDL_PSO::set_calibration(int calib_type_in, np::ndarray c_params_in, np::ndarray window_size_in, int N_cam_in) {
	Calib_type = calib_type_in;
	N_cam = N_cam_in;
	if (Calib_type==0) {
		c_params.reset();
		c_params.zeros(11,N_cam);
		window_size.reset();
		window_size.zeros(N_cam*2);
		N_pixels = 0;
		for (int n=0; n<N_cam; n++) {
			// window size
			window_size(n*2) = (int) p::extract<double>(window_size_in[n][0]);
			window_size(n*2+1) = (int) p::extract<double>(window_size_in[n][1]);
			N_pixels += window_size(n*2)*window_size(n*2+1);
			// c_params
			c_params(0,n) = p::extract<double>(c_params_in[0][n]);
			c_params(1,n) = p::extract<double>(c_params_in[1][n]);
			c_params(2,n) = p::extract<double>(c_params_in[2][n]);
			c_params(3,n) = p::extract<double>(c_params_in[3][n]);
			c_params(4,n) = p::extract<double>(c_params_in[4][n]);
			c_params(5,n) = p::extract<double>(c_params_in[5][n]);
			c_params(6,n) = p::extract<double>(c_params_in[6][n]);
			c_params(7,n) = p::extract<double>(c_params_in[7][n]);
			c_params(8,n) = p::extract<double>(c_params_in[8][n]);
			c_params(9,n) = p::extract<double>(c_params_in[9][n]);
			c_params(10,n) = p::extract<double>(c_params_in[10][n]);
		}
	}
	else if (Calib_type==1) {
		c_params.reset();
		c_params.zeros(16,N_cam);
		window_size.reset();
		window_size.zeros(N_cam*2);
		N_pixels = 0;
		for (int n=0; n<N_cam; n++) {
			// window size
			window_size(n*2) = (int) p::extract<double>(window_size_in[n][0]);
			window_size(n*2+1) = (int) p::extract<double>(window_size_in[n][1]);
			N_pixels += window_size(n*2)*window_size(n*2+1);
			// c_params
			c_params(0,n) = p::extract<double>(c_params_in[0][n]);
			c_params(1,n) = p::extract<double>(c_params_in[1][n]);
			c_params(2,n) = p::extract<double>(c_params_in[2][n]);
			c_params(3,n) = p::extract<double>(c_params_in[3][n]);
			c_params(4,n) = p::extract<double>(c_params_in[4][n]);
			c_params(5,n) = p::extract<double>(c_params_in[5][n]);
			c_params(6,n) = p::extract<double>(c_params_in[6][n]);
			c_params(7,n) = p::extract<double>(c_params_in[7][n]);
			c_params(8,n) = p::extract<double>(c_params_in[8][n]);
			c_params(9,n) = p::extract<double>(c_params_in[9][n]);
			c_params(10,n) = p::extract<double>(c_params_in[10][n]);
			c_params(11,n) = p::extract<double>(c_params_in[11][n]);
			c_params(12,n) = p::extract<double>(c_params_in[12][n]);
			c_params(13,n) = p::extract<double>(c_params_in[13][n]);
			c_params(14,n) = p::extract<double>(c_params_in[14][n]);
			c_params(15,n) = p::extract<double>(c_params_in[15][n]);
		}
	}
}

void MDL_PSO::set_batch_size(int N_batch_in) {
	N_batch = N_batch_in;
	positions.zeros(N_state,N_particles*N_batch);
	pb_positions.zeros(N_state,N_particles*N_batch);
	velocities.zeros(N_state,N_particles*N_batch);
	gb_positions.zeros(N_state,N_particles*N_batch);
	personal_best.zeros(N_components,N_particles*N_batch);
	global_best.zeros(N_components,N_batch);
	cost_mat.zeros(N_components,N_particles*N_batch);
	comp_indices.zeros(N_particles*N_batch);
	rand_ids.zeros(N_particles*N_batch);
	pb_update.zeros(N_state);
	rand_velocities.zeros(N_state,N_particles);
	cost_vec.resize(N_particles*N_batch);
	bin_img_body.clear();
	bin_img_wing.clear();
}

void MDL_PSO::set_bin_imgs(np::ndarray body_mask,np::ndarray wing_mask,int b_ind,int n_ind,int n_body,int n_wing) {
	int N_pix = 0;
	int i = 0;
	int j = 0;
	boost::dynamic_bitset temp_bitset1;
	boost::dynamic_bitset temp_bitset2;
	N_pix = window_size(n_ind*2)*window_size(n_ind*2+1);
	temp_bitset1.resize(N_pix);
	temp_bitset1.reset();
	for (int k=0; k<n_body; k++) {
		i = (int) p::extract<double>(body_mask[k][0]);
		j = (int) p::extract<double>(body_mask[k][1]);
		temp_bitset1.set(j*window_size(n_ind*2)+i);
	}
	temp_bitset2.resize(N_pix);
	temp_bitset2.reset();
	for (int k=0; k<n_wing; k++) {
		i = (int) p::extract<double>(wing_mask[k][0]);
		j = (int) p::extract<double>(wing_mask[k][1]);
		temp_bitset2.set(j*window_size(n_ind*2)+i);
	}
	bin_img_body.push_back(temp_bitset1);
	bin_img_wing.push_back(temp_bitset2);
}

arma::vec MDL_PSO::quat_SLERP(arma::vec qA, arma::vec qB, double t) {
	qA /= arma::norm(qA);
	qB /= arma::norm(qB);
	arma::vec qC(4);
	double Omega = arma::dot(qA,qB);
	double theta;
	if (Omega<0.0) {
		qA = -qA;
		Omega = -Omega;
	}
	if (Omega>0.9995) {
		qC = qA+t*(qB-qA);
		qC /= arma::norm(qC);
	}
	else {
		theta = acos(Omega);
		qC = (sin(theta*(1.0-t))/sin(theta))*qA+(sin(theta*t)/sin(theta))*qB;
		qC /= arma::norm(qC);
	}
	return qC;
}

arma::vec MDL_PSO::quat_diff(arma::vec qA, arma::vec qB) {
	arma::vec x_out(8);
	arma::Mat<double> QB_conj = {{qB(0),qB(1),qB(2),qB(3)},
		{qB(1),-qB(0),qB(3),-qB(2)},
		{qB(2),-qB(3),-qB(0),qB(1)},
		{qB(3),qB(2),-qB(1),-qB(0)}};
	x_out.rows(4,7) = arma::normalise(QB_conj*qA);
	if (x_out(4)<0.0) {
		x_out.rows(4,7) = -x_out.rows(4,7);
	}
	double theta = 0.0;
	double e_norm = 0.0;
	if (x_out(4)>0.999) {
		x_out(0) = 0.0;
		x_out(1) = 2.0*x_out(5);
		x_out(2) = 2.0*x_out(6);
		x_out(3) = 2.0*x_out(7);
	}
	else {
		theta = acos(x_out(4));
		if (theta>PI) {
			theta = PI;
		}
		e_norm = arma::norm(x_out.rows(5,7));
		x_out(0) = 0.0;
		x_out(1) = 2.0*theta*x_out(5)/e_norm;
		x_out(2) = 2.0*theta*x_out(6)/e_norm;
		x_out(3) = 2.0*theta*x_out(7)/e_norm;
	}
	return x_out;
}

arma::vec MDL_PSO::quat_multiply(arma::vec qA, arma::vec qB) {
	arma::Mat<double> QA = {{qA(0),-qA(1),-qA(2),-qA(3)},
		{qA(1),qA(0),qA(3),-qA(2)},
		{qA(2),-qA(3),qA(0),qA(1)},
		{qA(3),qA(2),-qA(1),qA(0)}};
	arma::vec q_out = MDL_PSO::quaternion_check(QA*qB);
	return q_out;
}

arma::vec MDL_PSO::transform_matrix(arma::vec sA, arma::vec tB, int trans) {
	arma::vec tC = {tB(0),tB(1),tB(2),1.0};
	arma::Mat<double> M_A(4,4);
	if (trans==0) {
		M_A = {{2.0*pow(sA(0),2.0)-1.0+2.0*pow(sA(1),2.0),2.0*sA(1)*sA(2)-2.0*sA(0)*sA(3),2.0*sA(1)*sA(3)+2.0*sA(0)*sA(2),sA(4)},
			{2.0*sA(1)*sA(2)+2.0*sA(0)*sA(3),2.0*pow(sA(0),2.0)-1.0+2.0*pow(sA(2),2.0),2.0*sA(2)*sA(3)-2.0*sA(0)*sA(1),sA(5)},
			{2.0*sA(1)*sA(3)-2.0*sA(0)*sA(2),2.0*sA(2)*sA(3)+2.0*sA(0)*sA(1),2.0*pow(sA(0),2.0)-1.0+2.0*pow(sA(3),2.0),sA(6)},
			{0.0,0.0,0.0,1.0}};
	}
	else if (trans==1) {
		M_A = {{2.0*pow(sA(0),2.0)-1.0+2.0*pow(sA(1),2.0),2.0*sA(1)*sA(2)+2.0*sA(0)*sA(3),2.0*sA(1)*sA(3)-2.0*sA(0)*sA(2),0.0},
			{2.0*sA(1)*sA(2)-2.0*sA(0)*sA(3),2.0*pow(sA(0),2.0)-1.0+2.0*pow(sA(2),2.0),2.0*sA(2)*sA(3)+2.0*sA(0)*sA(1),0.0},
			{2.0*sA(1)*sA(3)+2.0*sA(0)*sA(2),2.0*sA(2)*sA(3)-2.0*sA(0)*sA(1),2.0*pow(sA(0),2.0)-1.0+2.0*pow(sA(3),2.0),0.0},
			{0.0,0.0,0.0,1.0}};
	}
	else if (trans==2) {
		M_A = {{2.0*pow(sA(0),2.0)-1.0+2.0*pow(sA(1),2.0),2.0*sA(1)*sA(2)-2.0*sA(0)*sA(3),2.0*sA(1)*sA(3)+2.0*sA(0)*sA(2),0.0},
			{2.0*sA(1)*sA(2)+2.0*sA(0)*sA(3),2.0*pow(sA(0),2.0)-1.0+2.0*pow(sA(2),2.0),2.0*sA(2)*sA(3)-2.0*sA(0)*sA(1),0.0},
			{2.0*sA(1)*sA(3)-2.0*sA(0)*sA(2),2.0*sA(2)*sA(3)+2.0*sA(0)*sA(1),2.0*pow(sA(0),2.0)-1.0+2.0*pow(sA(3),2.0),0.0},
			{0.0,0.0,0.0,1.0}};
	}
	arma::vec t_out = M_A*tC;
	return t_out;
}

bool MDL_PSO::check_constraints(arma::vec state_in) {
	bool violations = false;
	arma::vec state_out = state_in;
	// Constraints:
	int c_type = 0;
	int p_ind  = 0;
	int c_ind  = 0;
	int jp_ind = 0;
	int jc_ind = 0;
	double dist = 0.0;
	arma::vec s_P(7);
	arma::vec s_JP(7);
	arma::vec s_C(7);
	arma::vec s_JC(7);
	arma::vec q_Pn(4);
	arma::vec s_Pn(7);
	arma::vec q_Cn(4);
	arma::vec s_Cn(7);
	arma::vec s_d(8);
	arma::vec q_d(4);
	arma::vec q_corr(4);
	double phi = 0.0;
	double theta = 0.0; 
	double eta = 0.0;
	double phi_corr = 0.0;
	double theta_corr = 0.0;
	double eta_corr = 0.0;
	arma::vec q_phi(4);
	arma::vec q_theta(4);
	arma::vec q_eta(4);
	arma::vec t_P(4);
	arma::vec t_JP(4);
	arma::vec t_Pn(4);
	arma::vec t_C(4);
	arma::vec t_JC(4);
	arma::vec t_Cn(4);
	arma::vec t_d(4);
	arma::vec t_corr(4);
	arma::vec q_goal(4);
	arma::vec t_goal(4);
	for (int k=0; k<N_constraints; k++) {
		c_type = constraint_types(k,0);
		p_ind  = constraint_types(k,1);
		c_ind  = constraint_types(k,2);
		jp_ind = constraint_types(k,3);
		jc_ind = constraint_types(k,4);
		s_P    = state_out.rows(state_comp(p_ind,0),state_comp(p_ind,0)+6);
		s_JP   = joint_locs.submat(0,jp_ind,6,jp_ind);
		s_C    = state_out.rows(state_comp(c_ind,0),state_comp(c_ind,0)+6);
		s_JC   = joint_locs.submat(0,jc_ind,6,jc_ind);
		q_Pn = MDL_PSO::quaternion_check(MDL_PSO::quat_multiply(s_P.rows(0,3),s_JP.rows(0,3)));
		t_Pn = MDL_PSO::transform_matrix(s_P,s_JP.rows(4,6),0);
		s_Pn.rows(0,3) = q_Pn;
		s_Pn.rows(4,6) = t_Pn.rows(0,2);
		q_Cn = MDL_PSO::quaternion_check(MDL_PSO::quat_multiply(s_C.rows(0,3),s_JC.rows(0,3)));
		t_Cn = MDL_PSO::transform_matrix(s_C,s_JC.rows(4,6),0);
		s_Cn.rows(0,3) = q_Cn;
		s_Cn.rows(4,6) = t_Cn.rows(0,2);
		s_d = MDL_PSO::quat_diff(q_Cn,q_Pn);
		q_d = MDL_PSO::quaternion_check(s_d.rows(4,7));
		// Compute translation vector
		t_d = MDL_PSO::transform_matrix(s_Pn,t_Cn.rows(0,2)-t_Pn.rows(0,2),1);
		// tx
		if (t_d(0)>constraint_bounds(7*k+3,1)*scale(p_ind)) {
			violations = true;
		}
		else if (t_d(0)<constraint_bounds(7*k+3,0)*scale(p_ind)) {
			violations = true;
		}
		// ty
		if (t_d(1)>constraint_bounds(7*k+4,1)*scale(p_ind)) {
			violations = true;
		}
		else if (t_d(1)<constraint_bounds(7*k+4,0)*scale(p_ind)) {
			violations = true;
		}
		// tz
		if (t_d(2)>constraint_bounds(7*k+5,1)*scale(p_ind)) {
			violations = true;
		}
		else if (t_d(2)<constraint_bounds(7*k+5,0)*scale(p_ind)) {
			violations = true;
		}
		// dist
		dist = arma::norm(t_d.rows(0,2));
		if (dist>constraint_bounds(7*k+6,1)*scale(p_ind)) {
			violations = true;
		}
		else if (dist<constraint_bounds(7*k+6,0)*scale(p_ind)) {
			violations = true;
		}
	}
	return violations;
}

arma::vec MDL_PSO::quaternion_check(arma::vec q_in) {
	arma::vec q_out = arma::normalise(q_in);
	if (q_out(0)<0.0) {
		q_out = -q_in;
	}
	return q_out;
}

void MDL_PSO::position_update(int N_neighbors,int iter_ind,int iter_max) {
	arma::uword i_neighbor;
	int i_ind = 0;
	int i_n = 0;
	int m0 = 0;
	int m1 = 0;
	int c_ind = 0;
	arma::vec q_j(4);
	arma::vec q_pb(4);
	arma::vec q_gb(4);
	arma::vec q_nb(4);
	arma::vec x_pb(8);
	arma::vec x_gb(8);
	arma::vec x_nb(8);
	arma::vec x_r1(8);
	arma::vec x_r2(8);
	arma::vec x_r3(8);
	arma::vec q_v(4);
	arma::Mat<double> Q_upd(4,4);
	arma::vec q_update(4);
	arma::vec t_j(3);
	arma::vec t_v(3);
	arma::vec t_pb(3);
	arma::vec t_gb(3);
	arma::vec t_nb(3);
	arma::vec t_update(3);
	double theta = 0.0;
	double xi_update;
	double w_iter = w*(0.5+0.5*((iter_max-iter_ind)*1.0)/(iter_max*1.0));
	double c3_iter = c3*((iter_max-iter_ind)*1.0)/(iter_max*1.0);
	for (int k=0; k<N_batch; k++) {
		for (int j=0; j<N_particles; j++) {
			// Random vectors:
			x_r1.randn();
			x_r2.randn();
			x_r3.randn();
			// Component nr
			i_ind = N_particles*k+j;
			c_ind = comp_indices(i_ind);
			m0 = state_comp(c_ind,0);
			m1 = state_comp(c_ind,1);
			// find best neighbor:
			if (N_batch>2*N_neighbors) {
				if (k>=N_neighbors && k<=(N_batch-N_neighbors-1)) {
					i_neighbor = global_best.submat(c_ind,k-N_neighbors,c_ind,k+N_neighbors).index_min();
					i_n = k-N_neighbors+i_neighbor;
				}
				else if (k<N_neighbors) {
					i_neighbor = global_best.submat(c_ind,0,c_ind,N_neighbors).index_min();
					i_n = i_neighbor;
				}
				else if (k>(N_batch-N_neighbors-1)) {
					i_neighbor = global_best.submat(c_ind,N_batch-N_neighbors-1,c_ind,N_batch-1).index_min();
					i_n = N_batch-1-N_neighbors+i_neighbor;
				}
			}
			else {
				i_neighbor = global_best.row(c_ind).index_min();
				i_n = i_neighbor;
			}
			// quaternion update:
			q_j  = positions.submat(m0,i_ind,m0+3,i_ind);
			q_pb = pb_positions.submat(m0,i_ind,m0+3,i_ind);
			q_gb = gb_positions.submat(m0,i_ind,m0+3,i_ind);
			q_nb = gb_positions.submat(m0,i_n,m0+3,i_n);
			x_pb = MDL_PSO::quat_diff(q_j,q_pb);
			x_gb = MDL_PSO::quat_diff(q_j,q_gb);
			x_nb = MDL_PSO::quat_diff(q_j,q_nb);
			velocities.submat(m0,i_ind,m0+3,i_ind) = w*velocities.submat(m0,i_ind,m0+3,i_ind)+c1*(x_r1.rows(0,3)%x_pb.rows(0,3))+c2*(x_r2.rows(0,3)%x_gb.rows(0,3)); //+c3_iter*(x_r3.rows(0,3)%x_nb.rows(0,3));
			theta = arma::norm(velocities.submat(m0+1,i_ind,m0+3,i_ind));
			if (theta<0.01) {
				q_v(0) = 1.0;
				q_v(1) = velocities(m0+1,i_ind)/2.0;
				q_v(2) = velocities(m0+2,i_ind)/2.0;
				q_v(3) = velocities(m0+3,i_ind)/2.0;
				q_v = arma::normalise(q_v);
			}
			else if (theta >= PI) {
				q_v(0) = -cos(theta/2.0);
				q_v(1) = -sin(theta/2.0)*velocities(m0+1,i_ind)/theta;
				q_v(2) = -sin(theta/2.0)*velocities(m0+2,i_ind)/theta;
				q_v(3) = -sin(theta/2.0)*velocities(m0+3,i_ind)/theta;
				q_v = arma::normalise(q_v);
			}
			else {
				q_v(0) = cos(theta/2.0);
				q_v(1) = sin(theta/2.0)*velocities(m0+1,i_ind)/theta;
				q_v(2) = sin(theta/2.0)*velocities(m0+2,i_ind)/theta;
				q_v(3) = sin(theta/2.0)*velocities(m0+3,i_ind)/theta;
				q_v = arma::normalise(q_v);
			}
			Q_upd = {{q_v(0),-q_v(1),-q_v(2),-q_v(3)},
				{q_v(1),q_v(0),-q_v(3),q_v(2)},
				{q_v(2),q_v(3),q_v(0),-q_v(1)},
				{q_v(3),-q_v(2),q_v(1),q_v(0)}};
			q_update = MDL_PSO::quaternion_check(Q_upd*q_j);
			positions.submat(m0,i_ind,m0+3,i_ind) = q_update;
			// translation update:
			t_j  = positions.submat(m0+4,i_ind,m0+6,i_ind);
			t_v  = velocities.submat(m0+4,i_ind,m0+6,i_ind);
			t_pb = pb_positions.submat(m0+4,i_ind,m0+6,i_ind);
			t_gb = gb_positions.submat(m0+4,i_ind,m0+6,i_ind);
			t_nb = gb_positions.submat(m0+4,i_n,m0+6,i_n);
			velocities.submat(m0+4,i_ind,m0+6,i_ind) = w*t_v+c1*(x_r1.rows(4,6)%(t_pb-t_j))+c2*(x_r2.rows(4,6)%(t_gb-t_j)); //+c3_iter*(x_r3.rows(4,6)%(t_nb-t_j));
			t_update = positions.submat(m0+4,i_ind,m0+6,i_ind)+velocities.submat(m0+4,i_ind,m0+6,i_ind);
			t_update(0) = clamp(t_update(0),state_bounds(m0+4,0),state_bounds(m0+4,1));
			t_update(1) = clamp(t_update(1),state_bounds(m0+5,0),state_bounds(m0+5,1));
			t_update(2) = clamp(t_update(2),state_bounds(m0+6,0),state_bounds(m0+6,1));
			positions.submat(m0+4,i_ind,m0+6,i_ind) = t_update;
			// bending update:
			if (m1>=(m0+7)) {
				velocities(m0+7,i_ind) = w*velocities(m0+7,i_ind)+c1*(x_r1(7)*(pb_positions(m0+7,i_ind)-positions(m0+7,i_ind)))+c2*(x_r2(7)*(gb_positions(m0+7,i_ind)-positions(m0+7,i_ind))); //+c3_iter*(x_r3(7)*(gb_positions(m0+7,i_n)-positions(m0+7,i_n)));
				xi_update = positions(m0+7,i_ind)+velocities(m0+7,i_ind);
				xi_update = clamp(xi_update,state_bounds(m0+7,0),state_bounds(m0+7,1));
				positions(m0+7,i_ind) = xi_update;
			}
		}
	}
}

arma::Mat<double> MDL_PSO::compute_state(arma::vec x) {
	bool ref_multiply = true;
	int k = 0;
	arma::Mat<double> Q_upd(4,4);
	arma::Mat<double> M_upd(4,4);
	double theta;
	double e_norm;
	arma::vec e_vec;
	e_vec.zeros(3);
	arma::vec q_p(4);
	arma::vec q_x(4);
	arma::vec t_x(4);
	arma::vec q_0(4);
	arma::vec t_0(4);
	arma::Mat<double> x_out;
	x_out.zeros(7,N_segments);
	for (int i=0; i<N_components; i++) {
		for (int j=0; j<N_segments; j++) {
			k = parent_child_ids(i,j);
			if (k>=0) {
				if (j==0) {
					x_out.col(k) = x.rows(state_conn(0,k),state_conn(6,k));
					x_out.submat(0,k,3,k) /= arma::norm(x_out.submat(0,k,3,k));
				}
				else {
					// q update:
					theta = state_const(0,k)*x(state_conn(0,k))/2.0;
					e_vec(0) = state_const(1,k);
					e_vec(1) = state_const(2,k);
					e_vec(2) = state_const(3,k);
					q_x = {cos(theta),e_vec(0)*sin(theta),e_vec(1)*sin(theta),e_vec(2)*sin(theta)};
					q_x /= arma::norm(q_x);
					x_out(0,k) = q_x(0);
					x_out(1,k) = q_x(1);
					x_out(2,k) = q_x(2);
					x_out(3,k) = q_x(3);
					x_out(4,k) = state_const(4,k)*scale(i);
					x_out(5,k) = state_const(5,k)*scale(i);
					x_out(6,k) = state_const(6,k)*scale(i);
				}
			}
		}
	}
	return x_out;
}

void MDL_PSO::setup_threads(string model_loc) {
	// Change directory
	chdir(model_loc.c_str());
	// Setup threads:
	cost_threads.clear();
	for (int k=0; k<N_batch; k++) {
		for (int j=0; j<N_particles; j++) {
			Cost temp_cost = Cost();
			temp_cost.set_calibration(Calib_type,c_params,window_size,N_cam);
			temp_cost.load_stl_files(file_names);
			temp_cost.set_scale(scale,parent_child_ids,body_or_wing);
			cost_threads.push_back(temp_cost);
		}
	}
}

void MDL_PSO::set_particle_start(np::ndarray init_state_in) {
	init_state.zeros(N_state,N_batch);
	// set init state
	for (int i=0; i<N_batch; i++) {
		for (int j=0; j<N_state; j++) {
			init_state(j,i) = p::extract<double>(init_state_in[j][i]);
		}
	}
}

np::ndarray MDL_PSO::PSO_fit() {
	// Initial positions and velocities:
	for (int k=0; k<N_batch; k++) {
		rand_velocities.randn();
		velocities.cols(k*N_particles,(k+1)*N_particles-1) = arma::repmat(state_dev,1,N_particles)%rand_velocities;
		//positions.cols(k*N_particles,(k+1)*N_particles-1) = arma::repmat(init_state.col(k),1,N_particles);
		for (int j=0; j<N_particles; j++) {
			if ((k-2)>=0 && (k+2)<N_batch) {
				positions.col(k*N_particles+j) = init_state.col(k-2+j%5);
			}
			else {
				positions.col(k*N_particles+j) = init_state.col(k);
			}
		}
		if (N_particles>5) {
			velocities.cols(k*N_particles,k*N_particles+5).zeros(); // Let first 5 particles stay at the initial position estimate
		}
		else {
			velocities.col(k*N_particles).zeros(); // Let first particle stay at the initial position estimate
		}
		pb_positions.cols(k*N_particles,(k+1)*N_particles-1) = positions.cols(k*N_particles,(k+1)*N_particles-1);
		gb_positions.cols(k*N_particles,(k+1)*N_particles-1) = positions.cols(k*N_particles,(k+1)*N_particles-1);
	}
	// Set cost matrices:
	personal_best.fill(1.0e9);
	global_best.fill(1.0e9);
	cost_mat.fill(1.0e9);
	// Set threads:
	for (int k=0; k<N_batch; k++) {
		temp_img_body.clear();
		temp_img_wing.clear();
		for (int n=0; n<N_cam; n++) {
			temp_img_body.push_back(bin_img_body.at(k*N_cam+n));
			temp_img_wing.push_back(bin_img_wing.at(k*N_cam+n));
		}
		for (int j=0; j<N_particles; j++) {
			cost_threads[k*N_particles+j].set_scale(scale,parent_child_ids,body_or_wing);
			cost_threads[k*N_particles+j].set_COM(COM.col(k));
			cost_threads[k*N_particles+j].set_bin_imgs(temp_img_body,temp_img_wing);
		}
	}
	// Advance particles with initial velocities:
	for (int k=0; k<N_components; k++) {
		comp_indices.fill(k);
		MDL_PSO::position_update(3,0,N_iter);
	}
	// Set component index:
	rand_ids.randu();
	for (int j=0; j<(N_particles*N_batch); j++) {
		//comp_indices(j) = rand_ids(j)*N_components;
		comp_indices(j) = 3+rand_ids(j)*2;
	}
	int m = 0;
	for (int i=0; i<N_iter; i++) {
		MDL_PSO::position_update(3,0,N_iter);
		for (int j=0; j<N_particles*N_batch; j++) {
			state_j = MDL_PSO::compute_state(positions.col(j));
			cost_threads[j].set_state(state_j);
			cost_threads[j].set_comp_ind(comp_indices(j));
		}
		// Compute cost:
		auto cost_func = [](Cost f_cost) {return f_cost.compute_cost();};
		transform(
			execution::par, 
			begin(cost_threads), 
			end(cost_threads), 
			begin(cost_vec), 
			cost_func
		);
		// Update personal and global best:
		for (int k=0; k<N_batch; k++) {
			for (int j=0; j<N_particles; j++) {
				m = comp_indices(k*N_particles+j);
				cost_mat(m,N_particles*k+j) = cost_vec[N_particles*k+j](m,0);
				if (cost_mat(m,N_particles*k+j)<personal_best(m,N_particles*k+j)) {
					personal_best(m,k*N_particles+j) = cost_mat(m,N_particles*k+j);
					pb_update = pb_positions.col(k*N_particles+j);
					pb_update.rows(state_comp(m,0),state_comp(m,1)) = positions.submat(state_comp(m,0),k*N_particles+j,state_comp(m,1),k*N_particles+j);
					//if (MDL_PSO::check_constraints(pb_update)==false) {
						pb_positions.submat(state_comp(m,0),k*N_particles+j,state_comp(m,1),k*N_particles+j) = pb_update.rows(state_comp(m,0),state_comp(m,1));
						if (cost_mat(m,N_particles*k+j)<global_best(m,k)) {
							global_best(m,k) = cost_mat(m,N_particles*k+j);
							gb_positions.submat(state_comp(m,0),k*N_particles,state_comp(m,1),(k+1)*N_particles-1) = arma::repmat(pb_update.rows(state_comp(m,0),state_comp(m,1)),1,N_particles);
						}
					//}
				}
			}
		}
		// Set component index:
		if (i%3==0) {
			rand_ids.randu();
			for (int j=0; j<(N_particles*N_batch); j++) {
				comp_indices(j) = rand_ids(j)*N_components;
				//comp_indices(j) = 3+rand_ids(j)*2;
			}
		}
		else {
			for (int j=0; j<(N_particles*N_batch); j++) {
				if (cost_mat(3,j)>cost_mat(4,j)) {
					comp_indices(j) = 3;
				}
				else {
					comp_indices(j) = 4;
				}
			}
		}
	}
	// Return optimized state and cost:
	state_best.zeros(N_state,N_batch);
	p::tuple shape = p::make_tuple(N_state+N_components,N_batch);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray state_out = np::zeros(shape,dtype);
	for (int k=0; k<N_batch; k++) {
		state_j = gb_positions.col(k*N_particles);
		state_best.col(k) = state_j;
		for (int j=0; j<(N_state+N_components); j++) {
			if (j<N_state) {
				state_out[j][k] = state_j(j);
			}
			else {
				state_out[j][k] = global_best(j-N_state,k);
			}
		}
	}
	return state_out;
}


np::ndarray MDL_PSO::return_model_img() {

	arma::Mat<double> state_i = MDL_PSO::compute_state(state_best.col(0));

	p::tuple shape = p::make_tuple(window_size(0),window_size(1),N_cam);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray img_out = np::zeros(shape,dtype);


	int pix_id = 0;
	int N_pix = 0;
	for (int k=0; k<N_components; k++) {
		cost_threads[0].set_comp_ind(k);
		cost_threads[0].set_state(state_i);
		cost_threads[0].compute_cost();
		for (int n=0; n<N_cam; n++) {
			N_pix = window_size(n*2)*window_size(n*2+1);
			if (k==0) {
				for (int i=0; i<window_size(n*2); i++) {
					for (int j=0; j<window_size(n*2+1); j++) {
						pix_id =window_size(n*2+1)*i+j;
						if (cost_threads[0].wing_image.at(n).test(pix_id)) {
							img_out[i][j][n] = 128.0;
						}
						if (cost_threads[0].body_image.at(n).test(pix_id)) {
							img_out[i][j][n] = 128.0;
						}
					}
				}
			}
			for (int i=0; i<window_size(n*2); i++) {
				for (int j=0; j<window_size(n*2+1); j++) {
					pix_id =window_size(n*2+1)*i+j;
					if (cost_threads[0].mdl_image.at(k*N_cam+n).test(pix_id)) {
						img_out[i][j][n] = 255.0;
					}
				}
			}
		}
	}
	return img_out;
}