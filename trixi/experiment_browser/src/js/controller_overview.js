var search_dict = {};
var removed_list = [];

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
                       <a role="button" class="remove-experiments btn btn-danger" style="float: left; margin-left: 2px;">Remove</a>');
        },
        columnDefs: [
            { targets: [0, 1, 2, 3], visible: true },
            { targets: "_all", visible: false }
        ]
    });

    // Compare experiments
    $('a.compare-experiments').on('click', function (e) {
        e.preventDefault();
        var experiments = [];
        $.each($('#overviewTable tr.selected'), function () {
            experiments.push($(this).find('th').eq(0).text());
        });
        if (experiments.length > 0) {
            var new_location = "experiment?";
            $.each(experiments, function (idx, val) {
                new_location = new_location + "exp=" + val + "&";
            });
            window.location.href = new_location;
        }
    });

    $('a.remove-experiments').on('click', function (e) {
        e.preventDefault();
        var experiments = [];
        $.each($('#overviewTable tr.selected'), function () {
            experiments.push($(this).find('th').eq(0).text());
        });
        if (experiments.length > 0) {
            var url_exp_remove = "experiment_remove?";
            $.each(experiments, function (idx, val) {
                url_exp_remove = url_exp_remove + "exp=" + val + "&";
                removed_list.push(val);
            });

            $.get(
                url_exp_remove,
                {},
                function (data) {
                    $.each(experiments, function (idx, val) {
                        removed_list.push(val);
                    });
                }
            );
            table.draw();

        }
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
            if (i != n-1) {
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
            if (i != n-1) {
                toggle($(obj), false);
            } else {
                toggle($(obj));
            }
        });
    });

    // Extended Search so you can search with <, <= , >, >=
    $.fn.dataTable.ext.search.push(
        function (settings, data, dataIndex) {

            var show_draw_row = true;

            if (removed_list.indexOf(data[0]) >= 0) {
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