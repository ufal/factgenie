
function updateConfig() {
    const config = {
        logging: {
            flask_debug: $('#debug').is(':checked'),
            level: $('#logging_level').val(),
        },
        host_prefix: $('#host_prefix').val(),
        login: {
            active: $('#login_active').is(':checked'),
            username: $('#login_username').val(),
            password: $('#login_password').val()
        }
    };

    $.post({
        url: `${url_prefix}/update_config`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify(config),
        success: function (response) {
            alert('Configuration updated successfully');
        },
        error: function (error) {
            alert('Error updating configuration: ' + error.responseText);
        }
    });

}


$(document).ready(function () {
    $("#show_hide_password a").on('click', function (event) {
        event.preventDefault();
        if ($('#show_hide_password input').attr("type") == "text") {
            $('#show_hide_password input').attr('type', 'password');
            $('#show_hide_password i').addClass("fa-eye-slash");
            $('#show_hide_password i').removeClass("fa-eye");
        } else if ($('#show_hide_password input').attr("type") == "password") {
            $('#show_hide_password input').attr('type', 'text');
            $('#show_hide_password i').removeClass("fa-eye-slash");
            $('#show_hide_password i').addClass("fa-eye");
        }
    });
});