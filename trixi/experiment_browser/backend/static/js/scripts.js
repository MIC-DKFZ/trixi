var slideIndex = {};

function plusSlides(tag, n) {
    n = parseInt(n);
    slideIndex[tag] += n;
    showSlides(tag);
}

function currentSlide(tag, n) {
    n = parseInt(n);
    slideIndex[tag] = n;
    showSlides(tag);
}


function loadImageLazily(img) {
    if (img) {
        if (img.getAttribute("lazysrc")) {
            if (!img.getAttribute("src")) {
                img.setAttribute("src", img.getAttribute("lazysrc"));
            }
        }
    }
}


function showSlides(tag) {

    if (!(tag in slideIndex)) {
        slideIndex[tag] = 0
    }

    var part = document.getElementById(tag);

    var i;
    var slides = part.getElementsByClassName("mySlides");
    var dots = part.getElementsByClassName("dot");
    var number = part.getElementsByClassName("number")[0];

    if (slideIndex[tag] >= slides.length) {
        slideIndex[tag] = 0;
    }
    if (slideIndex[tag] < 0) {
        slideIndex[tag] = slides.length - 1;
    }
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }
    for (i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" b_active", "");
    }


    slides[slideIndex[tag]].style.display = "block";

    var img = part.getElementsByClassName("mySlides")[slideIndex[tag]].getElementsByClassName("lazyimg")[0];
    loadImageLazily(img);

    dots[slideIndex[tag]].className += " b_active";

    number.value = slideIndex[tag] + 1
}

function numberChange(tag) {
    var part = document.getElementById(tag);
    var number = part.getElementsByClassName("number")[0];
    console.log(number.value);
    slideIndex[tag] = parseInt(number.value) - 1;
    showSlides(tag);

}

function alignImages() {

    var x = document.getElementById("imageAlignButton").checked;
    var imageSpaces = document.getElementsByClassName("imageSpace");
    var imgmaindiv = document.getElementById("theallimagediv");
    var imgcontentdivs = document.getElementsByClassName("theimagecontentdiv");

    if (x == true) {
        for (i = 0; i < imageSpaces.length; i++) {
            imageSpaces[i].style.display = 'inline-block';
        }
        imgmaindiv.style.overflowX = "scroll";
        for (i = 0; i < imgcontentdivs.length; i++) {
            imgcontentdivs[i].style.overflowX = 'visible';
        }
    }
    else {
        for (i = 0; i < imageSpaces.length; i++) {
            imageSpaces[i].style.display = 'none';
        }
        imgmaindiv.style.overflowX = "visible";
        for (i = 0; i < imgcontentdivs.length; i++) {
            imgcontentdivs[i].style.overflowX = 'scroll';
        }
    }

    console.log(imgcontentdivs);

}

function toggleliveplots() {

    var x = document.getElementById("toggleliveplots").checked;
    var liveplotdiv = document.getElementById("liveplots");
    var imgplotdiv = document.getElementById("imgplots");

    if (x === true) {
        liveplotdiv.style.display='block';
        imgplotdiv.style.display='none';
    }
    else {
        liveplotdiv.style.display='none';
        imgplotdiv.style.display='block';
    }

}