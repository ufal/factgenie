const statistics = window.statistics;
const ann_counts = statistics.ann_counts;

$(document).ready(function () {
    function populateTable(tableId, data, columns) {
        var table = $(`#${tableId} tbody`);
        table.empty(); // Clear existing rows

        for (const obj of data) {
            let row = '<tr>';
            for (const col of columns) {
                if (col === 'annotation_type') {
                    row += `<td>
                        <span class="badge" style="background-color: ${metadata.config.annotation_span_categories[obj[col]].color}; color: rgb(60, 65, 73);">
                            ${metadata.config.annotation_span_categories[obj[col]].name}
                        </span>
                    </td>`;
                } else {
                    row += `<td>${obj[col]}</td>`;
                }
            }
            row += '</tr>';
            table.append(row);
        }
        $(`#${tableId}`).bootstrapTable();
    }

    const fullTableColumns = ['dataset', 'split', 'setup_id', 'example_count', 'annotation_type', 'ann_count', 'avg_count', 'prevalence'];
    const spanTableColumns = ['example_count', 'annotation_type', 'ann_count', 'avg_count', 'prevalence'];
    const setupTableColumns = ['setup_id', 'example_count', 'annotation_type', 'ann_count', 'avg_count', 'prevalence'];
    const datasetTableColumns = ['dataset', 'split', 'example_count', 'annotation_type', 'ann_count', 'avg_count', 'prevalence'];

    populateTable('full-table', ann_counts.full, fullTableColumns);
    populateTable('span-table', ann_counts.span, spanTableColumns);
    populateTable('setup-table', ann_counts.setup, setupTableColumns);
    populateTable('dataset-table', ann_counts.dataset, datasetTableColumns);
});