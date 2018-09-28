import 'bootstrap'
import 'bootstrap-css'
import 'bootstrap-select'
import '../style/app.scss'
import 'material-icons/iconfont/material-icons.css'
import { library, dom } from '@fortawesome/fontawesome-svg-core'
import { faCircleNotch} from '@fortawesome/free-solid-svg-icons'
library.add(faCircleNotch)
import {dataTable} from 'datatables.net-dt'

const $ = require('jquery')

var search_dict = {};
var removed_list = [];
var last_td;

// Extended Search so you can search with <, <= , >, >=
$.fn.dataTable.ext.search.push(
    function (settings, data, dataIndex) {
        var show_draw_row = true;
        if (removed_list.indexOf($.trim(data[0])) >= 0) {
            return false;
        }

        for (var key in search_dict) {
            if (search_dict.hasOwnProperty(key)) {
                var val = search_dict[key];
                var data_el = data[key];

                if (data_el === "-") {
                    show_draw_row = false;
                }
                else if (val.startsWith(">=")) {
                    var numb = parseFloat(val.substring(2));
                    if (data_el < numb) {
                        show_draw_row = false;
                    }
                }
                else if (val.startsWith(">")) {
                    var numb = parseFloat(val.substring(1));
                    if (data_el <= numb) {
                        show_draw_row = false;
                    }
                }
                else if (val.startsWith("<=")) {
                    var numb = parseFloat(val.substring(2));
                    if (data_el > numb) {
                        show_draw_row = false;
                    }
                }
                else if (val.startsWith("<")) {
                    var numb = parseFloat(val.substring(1));
                    if (data_el >= numb) {
                        show_draw_row = false;
                    }
                }
            }
        }
        return show_draw_row;
    }
);

$(document).ready(function () {
    // Setup - add a text input to each footer cell
    $('#overviewTable tfoot th').each(function () {
        var title = $(this).text();
        $(this).html('<input type="text" placeholder="Search ' + title + '" />');
    });

    // Selectable rows
    $('#overviewTable tbody').on('click', 'tr', function (e) {
        // e.preventDefault();
        $(this).toggleClass('selected');
    });

    // DataTable
    var table = $('#overviewTable').DataTable({
        scrollY: '70vh',
        scrollCollapse: true,
        paging: false,
        scrollX: true,
        select: {
            items: 'rows'
        },
        info: "",
        dom: 'l<"toolbar">frtip',
        initComplete: function () {
            $("div.toolbar")
                .html('<a role="button" class="compare-experiments btn btn-primary" style="float: left">Details</a>\
                       <a role="button" class="combine-experiments btn btn-success" style="float: left;margin-left:                                                       2px;">Combine</a>\
                       <a role="button" class="remove-experiments btn btn-danger" style="float: left; margin-left: 2px;">Remove</a>');
        },
        columnDefs: [
            {targets: [0, 1, 2, 3, 4], visible: true},
            {targets: "_all", visible: false}
        ],
        aaSorting: [[2, 'desc']]
    });

    // Compare experiments
    $('a.compare-experiments').on('click', function (e) {
        e.preventDefault();
        var experiments = [];
        $.each($('#overviewTable tr.selected'), function () {
            var dir_name = $(this)[0].getAttribute("dir_name");
            experiments.push(dir_name);
        });
        if (experiments.length > 0) {
            var new_location = "experiment?";
            $.each(experiments, function (idx, val) {
                new_location = new_location + "exp=" + val + "&";
            });
            // window.location.href = new_location;
            window.open(new_location, '_blank');
        }
    });

    $('a.remove-experiments').on('click', function (e) {
        e.preventDefault();
        var experiments = {};
        $.each($('#overviewTable tr.selected'), function () {
            var dir_name = $(this)[0].getAttribute("dir_name");
            experiments[dir_name] = $.trim($(this).find('th').eq(0).text());
        });
        if (Object.keys(experiments).length > 0) {
            var url_exp_remove = "experiment_remove?";
            $.each(experiments, function (key, val) {
                url_exp_remove = url_exp_remove + "exp=" + key + "&";
                removed_list.push(val);
            });

            $.get(
                url_exp_remove,
                {},
                function (data) {
                    $.each(experiments, function (key, val) {
                        removed_list.push(val);
                    });
                }
            );
            table.draw();

        }
    });

    $('a.combine-experiments').on('click', function (e) {
        e.preventDefault();
        var experiments = [];
        $.each($('#overviewTable tr.selected'), function () {
            var dir_name = $(this)[0].getAttribute("dir_name");
            experiments.push(dir_name);
        });
        if (experiments.length > 1) {
            $('#combineModal').modal({show: true});
        }
        else if(experiments.length < 1){
            $('#combineModal2').modal({show: true});
        }
    });

    $('a.star_element').on('click', function (e) {
        e.preventDefault();

        var dir_name = $(this)[0].getAttribute("dir_name");
        var url_exp_star = "experiment_star?" + "exp=" + dir_name;
        var text = $(this)[0].text;
        var th = $(this);
        if (text === "star_border") {
            url_exp_star = url_exp_star + "&star=" + 1;
            $.get(
                url_exp_star,
                {},
                function () {

                    th[0].innerHTML = '<i class="material-icons"\n' +
                        '                                   style="vertical-align: middle">star</i>';
                }
            );
        }
        else if (text === "star") {
            url_exp_star = url_exp_star + "&star=" + 0;
            $.get(
                url_exp_star,
                {},
                function () {
                    th[0].innerHTML = '<i class="material-icons"\n' +
                        '                                   style="vertical-align: middle">star_border</i>';
                }
            );
        }

    });

    $('a.edit_name').on('click', function (e) {
        e.preventDefault();
        last_td = $(this)[0].parentElement.children[0];
    });

    $('#renameModal').on('show.bs.modal', function (event) {
        var button = $(event.relatedTarget); // Button that triggered the modal
        var name = button.data('name'); // Extract info from data-* attributes
        var dir = button.data('dir');

        var modal = $(this);
        modal.find('#rename-modal-button')[0].setAttribute("exp-dir", dir);
        modal.find('.modal-body input').val(name);
    });

    $('#rename-modal-button').on('click', function (e) {
        e.preventDefault();
        var name_input = document.getElementById('experiment-rename-input');
        var new_name = name_input.value;
        var exp_dir = $(this)[0].getAttribute("exp-dir");

        var rename_url = "experiment_rename?exp=" + exp_dir + "&name=" + new_name;

        $.get(
            rename_url,
            {},
            function () {
                last_td.innerHTML = new_name;
            }
        );


    });

    $('#combi-modal-button').on('click', function (e) {
        e.preventDefault();
        var name_input = document.getElementById('combi-name-input');
        var new_name = name_input.value;

        var save_input = document.getElementById('combi-save-input');
        var do_save = save_input.checked;

        var experiments = [];
        $.each($('#overviewTable tr.selected'), function () {
            var dir_name = $(this)[0].getAttribute("dir_name");
            experiments.push(dir_name);
        });
        if (experiments.length > 0) {
            var new_location = "experiment?";
            $.each(experiments, function (idx, val) {
                new_location = new_location + "exp=" + val + "&";
            });
            new_location += "name=" + new_name + "&";
            new_location += "save=" + do_save+ "&";
            new_location += "combi=true";
            // window.location.href = new_location;

            window.open(new_location, '_blank');
        }

    });

     $('#combi-modal-button2').on('click', function (e) {
        e.preventDefault();
        var name_input = document.getElementById('combi-name-input');
        var new_name = name_input.value;

        var experiments = [];
        $.each($('#overviewTable tr'), function () {
            var dir_name = $(this)[0].getAttribute("dir_name");
            if(dir_name != null) {
                experiments.push(dir_name);
            }
        });

        var chkArray = [];
        $(".chk:checked").each(function() {
            chkArray.push($(this).val());
        });

        var new_location = "combine?";
        $.each(experiments, function (idx, val) {
            new_location = new_location + "exp=" + val + "&";
        });
        $.each(chkArray, function (idx, val) {
            new_location = new_location + "group=" + val + "&";
        });
        new_location += "name=" + new_name + "&";

        $.get(
            new_location,
            {},
            function (data) {
                console.log("Combined Experiments");
                $('#LoadingModal').modal().hide();
                if (data === "1") {
                    location.reload();
                }
                else{
                    $('#ErrorModal').modal({show: true});
                }
            }
        );
        $('#LoadingModal').modal({show: true});



    });

    function toggle(button, redraw=true) {
        // Get the column API object
        var column = table.column(button.attr('data-column'));
        // Toggle the visibility
        column.visible(!column.visible(), redraw);

        if (button.attr('data-type') == "config") {
            if (column.visible()) {
                button.removeClass("btn-outline-success");
                button.addClass("btn-success");
            } else {
                button.removeClass("btn-success");
                button.addClass("btn-outline-success");
            }
        }
        if (button.attr('data-type') == "result") {
            if (column.visible()) {
                button.removeClass("btn-outline-warning");
                button.addClass("btn-warning");
            } else {
                button.removeClass("btn-warning");
                button.addClass("btn-outline-warning");
            }
        }
    };

    $('a.toggle-vis').on('click', function (e) {
        e.preventDefault();
        toggle($(this));
    });

    $('a.toggle-config').on('click', function (e) {
        e.preventDefault();
        var configbuttons = $('a[data-type="config"]');
        var n = configbuttons.length;
        configbuttons.each(function (i, obj) {
            if (i != n - 1) {
                toggle($(obj), false);
            } else {
                toggle($(obj));
            }
        });
    });

    $('a.toggle-result').on('click', function (e) {
        e.preventDefault();
        var resultbuttons = $('a[data-type="result"]');
        var n = resultbuttons.length;
        resultbuttons.each(function (i, obj) {
            if (i != n - 1) {
                toggle($(obj), false);
            } else {
                toggle($(obj));
            }
        });
    });

    // Apply the search
    table.columns().every(function () {
        var that = this;
        $('input', this.footer()).on('keyup change', function () {

            var cidx = that[0];
            var pre_str = this.value.replace(/\s/g, "");

            if (pre_str.startsWith(">") || pre_str.startsWith("<")) {
                search_dict[cidx] = pre_str;
                table.draw();

            }
            else {
                if (!pre_str || 0 === pre_str.length) {
                    delete search_dict[cidx];
                    table.draw();
                }
                if (that.search() !== this.value) {
                    that
                        .search(this.value)
                        .draw();
                }
            }
        });
    });

});
