function detailFormatter(index, row) {
    const keys = Object.keys(row).filter(key => key.match(/^\d+$/));
    const key = keys[keys.length - 1];

    return row[key];
}

function detailFilter(index, row) {
    // for all key value pairs in row, check if we can make a jquery object out of the value
    // if we can, check if the text of the object is "finished"
    // if it is, return true

    for (const key in row) {
        const value = row[key];
        try {
            const $value = $(value);
            if ($value.text().trim() === "finished") {
                return true;
            }
        } catch (e) {
            // pass
        }
    }
}
