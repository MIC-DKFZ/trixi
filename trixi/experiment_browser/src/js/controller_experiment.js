/*
var log_dict = {};

$(document).ready(function () {

    $('a.log-file-link').on('click', function (e) {
        e.preventDefault();

        var th = $(this);

        var expname = $(this)[0].attributes.expname.value;
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
                {exp: expname, log: logfile},
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
*/