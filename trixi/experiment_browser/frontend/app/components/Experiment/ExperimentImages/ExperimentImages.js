// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';
import SlideShow from 'react-image-show'

class ExperimentImages extends React.Component {

  get_data() {
    var image_path = [];
    var experiments = [];

    try {
      image_path = this.props.images.img_path.experiments;
      experiments = this.props.images.imgs.experiments;
    } catch (e) {
      if (e instanceof TypeError) {
        // console.log("ExperimentImages not initialized yet")
      } else {
        console.log(e)
      }
    }

    return {"image_path": image_path, "experiments": experiments}
  }

  generate_image_slider(key, image_base_path, data) {
    // parse paths
    var base_url = "http://localhost:5000/serve_image";
    var image_paths = [];
    for (var i = 0; i < data.length; i++) {
      var image_path = image_base_path + "/" + data[i];
      var url = new URL(base_url);
      url.searchParams.append("image_path", image_path);
      image_paths.push(url.href);
    }

    // get slider
    return (
      <SlideShow
        key={key}
        images={image_paths}
        width="200px"
        imagesWidth="200px"
        imagesHeight="200px"
        imagesHeightMobile="56vw"
        indicators fixedImagesHeight
      />
    )
  }

  generate_image_views() {
    var image_path = this.get_data().image_path;
    var experiments = this.get_data().experiments;
    var sliders = [];
    for (var key in experiments) {
      var image_names = experiments[key];
      var slider = this.generate_image_slider(key, image_path, image_names);
      sliders.push(slider)
    }

    return sliders;
  }

  render() {
    return (
      <div>
        <h1>Images</h1>
        {this.generate_image_views()}
      </div>
    )
  }
}

export default ExperimentImages;
