#ifndef SMTBX_STRUCTURE_FACTORS_DIRECT_TABLE_BASED_H
#define SMTBX_STRUCTURE_FACTORS_DIRECT_TABLE_BASED_H

#include <smtbx/error.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <smtbx/structure_factors/direct/standard_xray.h>
#include <cctbx/miller/lookup_utils.h>
#include <cctbx/xray/scatterer_lookup.h>
#include <fstream>

namespace smtbx { namespace structure_factors { namespace table_based {

  template <typename FloatType>
  class table_data {
  public:
    typedef FloatType float_type;
    typedef std::complex<float_type> complex_type;
  protected:
    // rows in order of original hkl index -> scatterer contribution
    af::shared<std::vector<complex_type> > data_;
    af::shared<cctbx::miller::index<> > miller_indices_;
    bool use_ad;
    af::shared<sgtbx::rot_mx> rot_mxs_;
    bool expanded;
  public:
    table_data()
    : expanded(false),
      use_ad(false)
    {}
    af::shared<std::vector<complex_type> > const &data() const {
      return data_;
    }

    af::shared<sgtbx::rot_mx> const &rot_mxs() const {
      return rot_mxs_;
    }

    af::shared<cctbx::miller::index<> > const &miller_indices() const {
      return miller_indices_;
    }

    bool use_AD() const {
      return use_ad;
    }

    bool is_expanded() const {
      return expanded;
    }
  };

  template <typename FloatType>
  class table_reader : public table_data<FloatType> {
  public:
    typedef FloatType float_type;
    typedef std::complex<float_type> complex_type;
  private:
    typedef table_data<FloatType> parent_t;
    const uctbx::unit_cell& u_cell;

    void read(af::shared<xray::scatterer<float_type> > const& scatterers,
      const std::string &file_name)
    {
      using namespace std;
      typedef cctbx::xray::scatterer_id_5<FloatType, 16> scatterer_id_t;
      ifstream in_file(file_name.c_str());
      string line;
      vector<std::string> toks;
      size_t lc = 0;
      vector<size_t> sc_indices(scatterers.size());
      bool header_read = false, ids_read = false;
      while (std::getline(in_file, line)) {
        lc++;
        boost::trim(line);
        if (line.empty()) {
          break;
        }
        toks.clear();
        // is header?
        if (!header_read) {
          boost::split(toks, line, boost::is_any_of(":"));
          if (toks.size() < 2) {
            continue;
          }
          if (boost::iequals(toks[0], "scatterers") && !ids_read) {
            std::vector<std::string> stoks;
            boost::trim(toks[1]);
            boost::split(stoks, toks[1], boost::is_any_of(" "));
            SMTBX_ASSERT(stoks.size() == scatterers.size());
            map<string, size_t> sc_map;
            for (size_t sci = 0; sci < scatterers.size(); sci++) {
              sc_map[boost::to_upper_copy(scatterers[sci].label)] = sci;
            }
            for (size_t sci = 0; sci < scatterers.size(); sci++) {
              boost::to_upper(stoks[sci]);
              map<string, size_t>::iterator fsci = sc_map.find(stoks[sci]);
              SMTBX_ASSERT(fsci != sc_map.end());
              sc_indices[sci] = fsci->second;
            }
          }
          else if (boost::iequals(toks[0], "scatterer_ids")) {
            std::vector<std::string> stoks;
            cctbx::xray::scatterer_cart_lookup<FloatType> scatter_lookup(u_cell, scatterers);
            boost::trim(toks[1]);
            boost::split(stoks, toks[1], boost::is_any_of(" "));
            SMTBX_ASSERT(stoks.size() == scatterers.size());
            for (size_t sci = 0; sci < scatterers.size(); sci++) {
              std::stringstream ss;
              ss << std::hex << stoks[sci];
              uint64_t id_val;
              ss >> id_val;
              scatterer_id_t sc_id(id_val);
              size_t idx = scatter_lookup.index_of_fractional(sc_id.get_crd(), sc_id.get_z(), 0, 1e-2);
              SMTBX_ASSERT(idx != ~0);
              sc_indices[sci] = idx;
            }
            ids_read = true;
          }
          else if (boost::iequals(toks[0], "AD")) {
            boost::trim(toks[1]);
            parent_t::use_ad = boost::iequals(toks[1], "false");
          }
          else if (boost::iequals(toks[0], "Symm")) {
            boost::trim(toks[1]);
            if (boost::iequals(toks[1], "expanded")) {
              parent_t::expanded = true;
            }
            else {
              vector<std::string> symms_toks;
              boost::split(symms_toks, toks[1], boost::is_any_of(";"));
              for (size_t sti = 0; sti < symms_toks.size(); sti++) {
                boost::trim(symms_toks[sti]);
                if (symms_toks[sti].empty()) {
                  break;
                }
                vector<std::string> symm_toks;
                boost::split(symm_toks, symms_toks[sti], boost::is_any_of(" "));
                SMTBX_ASSERT(symm_toks.size() == 9);
                sgtbx::rot_mx rmx;
                for (size_t mei = 0; mei < 9; mei++) {
                  rmx[mei] = boost::lexical_cast<int>(symm_toks[mei]);
                }
                parent_t::rot_mxs_.push_back(rmx);
              }
            }
          }
          else if (boost::iequals(toks[0], "data")) {
            header_read = true;
          }
        }
        // data
        else {
          boost::split(toks, line, boost::is_any_of(" "));
          SMTBX_ASSERT(toks.size() == 3 + scatterers.size());
          cctbx::miller::index<> mi(
            atoi(toks[0].c_str()),
            atoi(toks[1].c_str()),
            atoi(toks[2].c_str()));
          parent_t::miller_indices_.push_back(mi);
          vector<complex_type> row;
          row.resize(scatterers.size());
#pragma omp parallel for
          for (int sci = 3; sci < toks.size(); sci++) {
            size_t ci = toks[sci].find(',');
            if (ci != string::npos) {
              complex_type v(
                  atof(toks[sci].substr(0, ci).c_str()),
                  atof(toks[sci].substr(ci + 1).c_str()));
              row[sc_indices[sci - 3]] = v;
            }
            else {
              row[sc_indices[sci - 3]] = complex_type(
                boost::lexical_cast<float_type>(toks[sci]));
            }
          }
          parent_t::data_.push_back(row);
        }
      }
    }
    
    void read_binary(af::shared<xray::scatterer<float_type> > const &scatterers,
      const std::string &file_name)
    {
      using namespace std;
      ifstream tsc_file(file_name.c_str(), ios::binary);

      const size_t charsize = sizeof(char);
      int head[1] = { 0 };
      const size_t intsize = sizeof(head);
      const size_t complex_doublesize = sizeof(complex<double>);
      const size_t complex_type_size = sizeof(complex_type);
      //If the size is not according to double type the binary will not be readable
      SMTBX_ASSERT(complex_doublesize == complex_type_size);
      tsc_file.read((char*)&head, intsize);
      const int nr_scat = scatterers.size();
      string header_str;
      if (head[0] != 0) {
        vector<char> header(head[0]);
        tsc_file.read(&header[0], head[0] * charsize);
        header_str = header.data();
      }
      //read scatterer labels and map onto scattterers list
      int sc_len[1] = {0};
      tsc_file.read((char*)&sc_len, intsize);
      vector<char> scat_line(sc_len[0]);
      tsc_file.read((char*)scat_line.data(), sc_len[0] * charsize);
      string scat_str(scat_line.begin(),scat_line.end());
      vector<string> toks;
      boost::split(toks, scat_str, boost::is_any_of(" "));
      SMTBX_ASSERT(toks.size() == nr_scat);
      map<string, size_t> sc_map;
      for (size_t sci = 0; sci < nr_scat; sci++) {
        sc_map[boost::to_upper_copy(scatterers[sci].label)] = sci;
      }
      vector<size_t> sc_indices(nr_scat);
      for (size_t sci = 0; sci < nr_scat; sci++) {
        boost::to_upper(toks[sci]);
        map<string, size_t>::iterator fsci = sc_map.find(toks[sci]);
        SMTBX_ASSERT(fsci != sc_map.end())("scatterer " + toks[sci] + " not found!");
        sc_indices[sci] = fsci->second;
      }
      SMTBX_ASSERT(sc_map.size() == scatterers.size());
      //binary tsc files will only be written in expanded mode
      parent_t::expanded = true;
      //read number of indices in tscb file
      int nr_hkl[1] = { 0 };
      tsc_file.read((char*)&nr_hkl, intsize);
      //read indices and scattering factors row by row
      int index[3] = { 0,0,0 };
      vector<complex<double> > row(nr_scat);
      parent_t::data_.reserve(nr_hkl[0] * nr_scat);
      for (int run = 0; run < *nr_hkl; run++) {
        tsc_file.read((char*)&index, 3*intsize);
        cctbx::miller::index<> mi(index[0], index[1], index[2]);
        parent_t::miller_indices_.push_back(mi);
        for (int i = 0; i < nr_scat; i++) {
          tsc_file.read((char*)&(row[sc_indices[i]]), complex_doublesize);
        }
        parent_t::data_.push_back(row);
      }
      tsc_file.close();
      SMTBX_ASSERT(!tsc_file.bad());
    }

  public:
    table_reader(const uctbx::unit_cell& u_cell,
      af::shared<xray::scatterer<float_type> > const &scatterers,
      const std::string &file_name)
      : u_cell(u_cell)
    {
      if(file_name.find(".tscb")!=std::string::npos){
        read_binary(scatterers, file_name);
      }
      else{
        read(scatterers, file_name);
      }
    }
  };

  template <typename FloatType>
  class table_based_isotropic
    : public direct::one_scatterer_one_h::scatterer_contribution<FloatType>
  {
    typedef direct::one_scatterer_one_h::scatterer_contribution<FloatType>
      base_type;
  public:
    typedef FloatType float_type;
    typedef std::complex<float_type> complex_type;
  private:
    miller::lookup_utils::lookup_tensor<float_type> mi_lookup;
    // hkl x scatterer x contribution
    af::shared <std::vector<complex_type> > data;
  public:
    // Copy constructor
    table_based_isotropic(const table_based_isotropic &tbsc)
      :
      mi_lookup(tbsc.mi_lookup),
      data(tbsc.data)
    {}

    table_based_isotropic(
      af::shared< xray::scatterer<float_type> > const &scatterers,
      table_reader<FloatType> const &data_,
      sgtbx::space_group const &space_group,
      bool anomalous_flag)
      :
      data(data_.miller_indices().size())
    {
      SMTBX_ASSERT(data_.rot_mxs().size() <= 1);
      for (size_t i = 0; i < data.size(); i++) {
        data[i].resize(scatterers.size());
        for (size_t j = 0; j < scatterers.size(); j++) {
          data[i][j] = data_.data()[i][j];
        }
      }
      mi_lookup = miller::lookup_utils::lookup_tensor<float_type>(
        data_.miller_indices().const_ref(),
        space_group,
        anomalous_flag);
    }

    virtual complex_type get(std::size_t scatterer_idx,
      miller::index<> const &h) const
    {
      long h_idx = mi_lookup.find_hkl(h);
      SMTBX_ASSERT(h_idx >= 0);
      return data[static_cast<size_t>(h_idx)][scatterer_idx];
    }

    virtual std::vector<complex_type> const &get_full(std::size_t scatterer_idx,
      miller::index<> const &h) const
    {
      SMTBX_NOT_IMPLEMENTED();
      throw 1;
    }

    virtual base_type &at_d_star_sq(
      float_type d_star_sq)
    {
      return *this;
    }

    virtual bool is_spherical() const {
      return true;
    }

    virtual base_type *raw_fork() const {
      return new table_based_isotropic(*this);
    }
  };

  template <typename FloatType>
  class table_based_anisotropic
    : public direct::one_scatterer_one_h::scatterer_contribution<FloatType>
  {
    typedef direct::one_scatterer_one_h::scatterer_contribution<FloatType>
      base_type;
  public:
    typedef FloatType float_type;
    typedef std::complex<float_type> complex_type;
  private:
    miller::lookup_utils::lookup_tensor<float_type> mi_lookup;
    // hkl x scatterer x hkl*r contribution
    af::shared <af::shared<std::vector<complex_type> > > data;
  public:
    // Copy constructor
    table_based_anisotropic(const table_based_anisotropic &tbsc)
      :
      mi_lookup(tbsc.mi_lookup),
      data(tbsc.data)
    {}

    table_based_anisotropic(
      af::shared< xray::scatterer<float_type> > const &scatterers,
      table_reader<FloatType> const &data_,
      sgtbx::space_group const &space_group,
      bool anomalous_flag)
    {
      SMTBX_ASSERT(data_.rot_mxs().size() == space_group.n_smx());
      SMTBX_ASSERT((data_.data().size() % space_group.n_smx()) == 0);

      std::vector<size_t> r_map;
      r_map.resize(space_group.n_smx());
      for (size_t i = 0; i < space_group.n_smx(); i++) {
        sgtbx::rot_mx const& r = data_.rot_mxs()[i];
        bool found = false;
        for (size_t mi = 0; mi < space_group.n_smx(); mi++) {
          if (r == space_group.smx(mi).r()) {
            r_map[i] = mi;
            found = true;
            break;
          }
        }
        SMTBX_ASSERT(found);
      }
      data.resize(data_.data().size() / space_group.n_smx());
      af::shared<cctbx::miller::index<> > lookup_indices(data.size());
      for (size_t hi = 0; hi < data.size(); hi++) {
        af::shared<std::vector<complex_type> > row(scatterers.size());
        for (size_t sci = 0; sci < scatterers.size(); sci++) {
          std::vector<complex_type> h_row(space_group.n_smx());
          for (size_t mi = 0; mi < space_group.n_smx(); mi++) {
            const size_t r_off = data.size() * mi;
            miller::index<> h =
              data_.miller_indices()[hi] * space_group.smx(r_map[mi]).r();
            SMTBX_ASSERT(h == data_.miller_indices()[r_off + hi]);
            h_row[r_map[mi]] = data_.data()[r_off + hi][sci];
          }
          row[sci] = h_row;
        }
        data[hi] = row;
        lookup_indices[hi] = data_.miller_indices()[hi];
      }

      mi_lookup = miller::lookup_utils::lookup_tensor<float_type>(
        lookup_indices.const_ref(),
        space_group,
        anomalous_flag);
    }

    virtual complex_type get(std::size_t scatterer_idx,
      miller::index<> const &h) const
    {
      SMTBX_NOT_IMPLEMENTED();
      throw 1;
    }

    virtual std::vector<complex_type> const &get_full(std::size_t scatterer_idx,
      miller::index<> const &h) const
    {
      long h_idx = mi_lookup.find_hkl(h);
      SMTBX_ASSERT(h_idx >= 0);
      return data[static_cast<size_t>(h_idx)][scatterer_idx];
    }


    virtual base_type &at_d_star_sq(
      float_type d_star_sq)
    {
      return *this;
    }

    virtual bool is_spherical() const {
      return false;
    }

    virtual base_type *raw_fork() const {
      return new table_based_anisotropic(*this);
    }
  };

  template <typename FloatType>
  class lookup_based_anisotropic
    : public direct::one_scatterer_one_h::scatterer_contribution<FloatType>
  {
    typedef direct::one_scatterer_one_h::scatterer_contribution<FloatType>
      base_type;
  public:
    typedef FloatType float_type;
    typedef std::complex<float_type> complex_type;
  private:
    typedef std::map<cctbx::miller::index<>, std::size_t,
      cctbx::miller::fast_less_than<> > lookup_t;
    lookup_t mi_lookup;
    sgtbx::space_group const &space_group;
    af::shared<std::vector<complex_type> > data;
    bool anomalous_flag;
    mutable std::vector<complex_type> tmp;
  public:
    // Copy constructor
    lookup_based_anisotropic(const lookup_based_anisotropic &lbsc)
      :
      mi_lookup(lbsc.mi_lookup),
      space_group(lbsc.space_group),
      data(lbsc.data),
      anomalous_flag(lbsc.anomalous_flag),
      tmp(lbsc.tmp.size())
    {}

    lookup_based_anisotropic(
      af::shared< xray::scatterer<float_type> > const &scatterers,
      table_reader<FloatType> const &data_,
      sgtbx::space_group const &space_group,
      bool anomalous_flag)
      :
      space_group(space_group),
      data(data_.miller_indices().size()),
      anomalous_flag(anomalous_flag),
      tmp(space_group.n_smx())
    {
      SMTBX_ASSERT(data_.rot_mxs().size() <= 1);
      SMTBX_ASSERT(data_.is_expanded());
      const af::shared<cctbx::miller::index<> > &indices = data_.miller_indices();
      for (size_t i = 0; i < data.size(); i++) {
        mi_lookup[indices[i]] = i;
        data[i].resize(scatterers.size());
#pragma omp parallel for
        for (int j = 0; j < scatterers.size(); j++) {
          data[i][j] = data_.data()[i][j];
        }
      }
    }
    // for testing
    lookup_based_anisotropic(
      uctbx::unit_cell const &unit_cell,
      sgtbx::space_group const &space_group,
      af::shared<xray::scatterer<float_type> > const &scatterers,
      direct::one_scatterer_one_h::isotropic_scatterer_contribution<FloatType> &isc,
      af::shared<cctbx::miller::index<> > const &indices)
      :
      space_group(space_group),
      data(indices.size()*space_group.n_smx()),
      tmp(space_group.n_smx())
    {
      for (size_t i = 0; i < indices.size(); i++) {
        float_type d_star_sq = unit_cell.d_star_sq(indices[i]);
        direct::one_scatterer_one_h::scatterer_contribution<FloatType> const& sc =
          isc.at_d_star_sq(d_star_sq);
        for (size_t j = 0; j < space_group.n_smx(); j++) {
          size_t d_off = j * indices.size() + i;
          miller::index<> h = indices[i] * space_group.smx(j).r();
          mi_lookup[h] = d_off;
          data[d_off].resize(scatterers.size());
          for (size_t k = 0; k < scatterers.size(); k++) {
            complex_type v = sc.get(k, h);
            if (scatterers[k].flags.use_fp_fdp()) { // revert of applied...
              v = complex_type(v.real() - scatterers[k].fp, 0);
            }
            data[d_off][k] = v;
          }
        }
      }
    }

    virtual complex_type get(std::size_t scatterer_idx,
      miller::index<> const &h) const
    {
      SMTBX_NOT_IMPLEMENTED();
      throw 1;
    }

    virtual std::vector<complex_type> const &get_full(std::size_t scatterer_idx,
      miller::index<> const &h_) const
    {
      for (std::size_t i = 0; i < space_group.n_smx(); i++) {
        miller::index<> h = h_ * space_group.smx(i).r();
        lookup_t::const_iterator l = mi_lookup.find(h);
        if (l == mi_lookup.end() && !anomalous_flag) {
          h *= -1;
          l = mi_lookup.find(h);
          SMTBX_ASSERT(l != mi_lookup.end())(h.as_string());
          tmp[i] = std::conj(data[l->second][scatterer_idx]);
        }
        else{
          SMTBX_ASSERT(l != mi_lookup.end())(h.as_string());
          tmp[i] = data[l->second][scatterer_idx];
        }
      }
      return tmp;
    }

    virtual base_type &at_d_star_sq(
      float_type d_star_sq)
    {
      return *this;
    }

    virtual bool is_spherical() const {
      return false;
    }

    virtual base_type *raw_fork() const {
      return new lookup_based_anisotropic(*this);
    }
  };

  template <typename FloatType>
  struct builder {
    static direct::one_scatterer_one_h::scatterer_contribution<FloatType> *
      build(
        const uctbx::unit_cell& u_cell,
        af::shared< xray::scatterer<FloatType> > const &scatterers,
        std::string const &file_name,
        sgtbx::space_group const &space_group,
        bool anomalous_flag)
    {
      table_reader<FloatType> data(u_cell, scatterers, file_name);
      if(!data.use_AD()){
        anomalous_flag = false;
      }
      if (data.rot_mxs().size() <= 1) {
        if (data.is_expanded()) {
          return new lookup_based_anisotropic<FloatType>(
            scatterers,
            data,
            space_group,
            anomalous_flag);
        }
        else {
          return new table_based_isotropic<FloatType>(
            scatterers,
            data,
            space_group,
            anomalous_flag);
        }
      }
      return new table_based_anisotropic<FloatType>(
        scatterers,
        data,
        space_group,
        anomalous_flag);
    }

    static direct::one_scatterer_one_h::scatterer_contribution<FloatType> *
      build_lookup_based_for_tests(
        uctbx::unit_cell const &unit_cell,
        sgtbx::space_group const &space_group,
        af::shared<xray::scatterer<FloatType> > const &scatterers,
        xray::scattering_type_registry const &scattering_type_registry,
        af::shared<cctbx::miller::index<> > const &indices)
    {
      direct::one_scatterer_one_h::isotropic_scatterer_contribution<FloatType>
        isc(scatterers, scattering_type_registry);
      return new lookup_based_anisotropic<FloatType>(
        unit_cell,
        space_group,
        scatterers,
        isc,
        indices);
    }
  };

}}} // smtbx::structure_factors::table_based

#endif // GUARD
