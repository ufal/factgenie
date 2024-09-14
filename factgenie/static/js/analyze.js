const statistics = window.statistics;

const ann_counts = statistics.ann_counts;
const ann_counts_full = ann_counts.full;

var table = $('#statistics-table tbody');

$(document).ready(function () {
    // for each object in ann_counts_full
    for (const obj of ann_counts_full) {
        const row = `<tr>
            <td>${obj.dataset}</td>
            <td>${obj.split}</td>
            <td>${obj.setup_id}</td>
            <td>${obj.annotation_type}</td>
            <td>${obj.ann_count}</td>
            <td>${obj.avg_count}</td>
        </tr>`;
        table.append(row);
    }
    $('#statistics-table').bootstrapTable();
});

