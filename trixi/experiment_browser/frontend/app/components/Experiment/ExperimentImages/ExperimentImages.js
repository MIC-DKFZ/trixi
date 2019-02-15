// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';
import SlideShow from 'react-image-show'

class ExperimentImages extends React.Component {
  generate_dummy_data() {
    this.data = [
      "https://quanlieu.github.io/react-image-show/dd0200001e3a8ab1bb8b43ce1bc91a95.png",
      "https://quanlieu.github.io/react-image-show/48ac2dcdcafc65d2b710e4bf626bd2a6.png",
      "https://quanlieu.github.io/react-image-show/6189b61d45aff551635c8c66ce258753.png",
      "https://quanlieu.github.io/react-image-show/305f030845ad6b159af134e15fe01eaf.png",
      "https://quanlieu.github.io/react-image-show/0f8e1d1ec9bcbcd8e91a7014a62da202.png",

    ]
  }

  generate_image_slider() {
    return (
      <SlideShow
        images={this.data}
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

  render() {
    this.generate_dummy_data();
    return (
      <div>
        <h1>Images</h1>
        {this.generate_image_slider()}
      </div>
      )
  }
}

export default ExperimentImages;
