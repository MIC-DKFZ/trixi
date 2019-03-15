// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';
import SlideShow from 'react-image-show'

class ExperimentImages extends React.Component {

  generate_image_slider(key, image_path, data) {
    // parse paths
    var image_paths = [];
    for(var i=0; i<data.length;i++) {
      var path = image_path + "/" + data[i];
      image_paths.push(path)
    }

    // Todo: load images from source (maybe backend). Simple display isn't working.

    // get slider
    return (
      <SlideShow
        key={key}
        images={image_paths}
        width="200px"
        imagesWidth="200px"
        imagesHeight="200px"
        imagesHeightMobile="56vw"
        // thumbnailsWidth="920px"
        // thumbnailsHeight="12vw"
        // thumbnails
        indicators fixedImagesHeight
      />
    )
  }

  generate_image_views() {
    var output = "";
    try {
      var image_path = this.props.images.img_path.experiments;
      var experiments = this.props.images.imgs.experiments;
      console.log(experiments)
      var sliders = [];
      for(var key in experiments) {
        var image_names = experiments[key];
        var slider = this.generate_image_slider(key, image_path, image_names)
        sliders.push(slider)
      }
      output = sliders
    }
    catch (e) {
      if (e instanceof TypeError) {
        console.log("not initialized yet...")
      } else {
        console.log(e)
      }
    }
    return output;
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
