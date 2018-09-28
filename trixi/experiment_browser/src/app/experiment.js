import 'bootstrap'
import 'bootstrap-css'
import 'plotly'
import 'popper.js'
import '../style/app.scss'

import 'material-icons/iconfont/material-icons.css'


const $ = require('jquery')


import { library, dom } from '@fortawesome/fontawesome-svg-core'
import { faCircleNotch} from '@fortawesome/free-solid-svg-icons'
library.add(faCircleNotch)

var log_dict = {};

// Replace any existing <i> tags with <svg> and set up a MutationObserver to
// continue doing this as the DOM changes.


$(document).ready(function () {
    dom.watch()

    $('a.log-file-link').on('click', function (e) {
        e.preventDefault();

        var th = $(this);

        var expname = $(this)[0].attributes.expname.value;
        var expdir = $(this)[0].attributes.expdir.value;
        var logfile = $(this)[0].attributes.logfile.value;

        $(".nav").find(".log-file-link-" + expname).removeClass("active");

        var target_elem = document.getElementById(expname + "-log-content");


        var dict_key = expname + "-" + logfile;
        if (dict_key in log_dict) {
            target_elem.innerHTML = log_dict[dict_key];
            th.addClass("active");

        }
        else {
            $.get(
                "experiment_log",
                {exp: expdir, log: logfile},
                function (data) {
                    console.log("request");
                    log_dict[dict_key] = [data];
                    target_elem.innerHTML = data;
                    th.addClass("active");
                }
            );
        }
        // toggle($(this));
    });

    $('a.log-file-close').on('click', function (e) {
        e.preventDefault();


        var expname = $(this)[0].attributes.expname.value;
        var target_elem = document.getElementById(expname + "-log-content");
        target_elem.innerHTML = "";

        $(".nav").find(".log-file-link-" + expname).removeClass("active");

    });

    $('a.star_element').on('click', function (e) {
        e.preventDefault();

        console.log("xD");

        var dir_name = $(this)[0].getAttribute("dir_name");
        var url_exp_star = "experiment_star?" + "exp=" + dir_name;
        var text = $.trim($(this)[0].text);
        var th = $(this);

        console.log(text);
        console.log(th);

        if (text === "star_border") {
            url_exp_star = url_exp_star + "&star=" + 1;
            $.get(
                url_exp_star,
                {},
                function () {

                    th[0].innerHTML = '<i class="material-icons">star</i>';
                }
            );
        }
        else if (text === "star") {
            url_exp_star = url_exp_star + "&star=" + 0;
            $.get(
                url_exp_star,
                {},
                function () {
                    th[0].innerHTML = '<i class="material-icons">star_border</i>';
                }
            );
        }

    });

    $('a.image_link').on('click', function (e) {
        e.preventDefault();
        console.log("Image clicked");
    });

    $('a.image_link').on('dblclick', function (e) {
        e.preventDefault();
        window.open($(this)[0].href, '_blank')
    });

    function findGetParameter(parameterName) {
        var result = null,
            tmp = [];
        location.search
            .substr(1)
            .split("&")
            .forEach(function (item) {
                tmp = item.split("=");
                if (tmp[0] === parameterName) result = decodeURIComponent(tmp[1]);
            });
        return result;
    }

    $.get(
        "experiment_plots?" + window.location.search.substr(1),
        {},
        function (data) {
            console.log("request");

            var graph = JSON.parse(data);
            var plothtml = "";

            var target_elem = document.getElementById("plotlyplots");

            $.each(graph["graphs"], function (index, value) {

                var height = 450 + 40 * graph["traces"][index];

                plothtml += '<div class="plot-box" style="height: ';
                plothtml += '' + height;
                plothtml += 'px">';
                plothtml += value;
                plothtml += '</div>';

            });

            target_elem.innerHTML = plothtml;


            $.each(graph["graphs"], function (index, value) {
                var js_strng = value.split('<script type="text/javascript">')[1];
                js_strng = js_strng.substring(0, js_strng.length - 9);
                eval(js_strng);
            });


        }
    );

});
