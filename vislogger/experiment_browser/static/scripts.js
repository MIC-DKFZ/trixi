var slideIndex = 0;
showSlides(slideIndex);

function plusSlides(n) {
    showSlides(slideIndex += n);
}

function currentSlide(n) {
    showSlides(slideIndex = n);
}


function loadImageLazily(img)
{
    if (img) {
        if (img.getAttribute("lazysrc")) {
            if (!img.getAttribute("src")) {
                img.setAttribute("src", img.getAttribute("lazysrc"));
            }
        }
    }
}


function showSlides(n) {
    var i;
    var slides = document.getElementsByClassName("mySlides");
    var dots = document.getElementsByClassName("dot");
    var number = document.getElementById("number");

    if (n >= slides.length) {
        slideIndex = 0
    }
    if (n < 0) {
        slideIndex = slides.length - 1
    }
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }
    for (i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" active", "");
    }

    console.log("New Index: ");
    console.log(n);
    console.log(slideIndex);

    slides[slideIndex].style.display = "block";

    var img = document.getElementsByClassName("mySlides")[slideIndex].getElementsByClassName("lazyimg")[0];
    loadImageLazily(img);

    dots[slideIndex].className += " active";

    number.setAttribute("value", slideIndex+1)
}

function numberChange() {
    var number = document.getElementById("number");
    console.log(number.value);
    showSlides(slideIndex = parseInt(number.value) - 1);

}