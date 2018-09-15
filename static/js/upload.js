$.fn.csv2arr = function (callback) {
    if (typeof(FileReader) == 'undefined') {    //if not H5
        alert("Your browser is too old,please use Chrome or Firefox");
        return false;
    }
    if (!$(this)[0].files[0]) {
        alert("Please select a file");
        return false;
    }
    var fReader = new FileReader();
    fReader.readAsDataURL($(this)[0].files[0]);
    $fileDOM = $(this);
    fReader.onload = function (evt) {
        var data = evt.target.result;
        var encoding = checkEncoding(data);
        Papa.parse($($fileDOM)[0].files[0], {
            encoding: encoding,
            complete: function (results) {
                var res = results.data;
                console.log("被选中的是：", $("input[name=select]:checked").val());
                if ($('input[name=select]:checked').val() == 'Normal') {
                    console.log("成功进入normal");
                    $.ajax({
                        url: '/data_workshop',
                        type: 'POST',
                        data: {json_data: JSON.stringify(res)},
                        success: function (data) {
                            if(data){
                                alert('Your file was uploaded sucessfully!');
                                window.location.href = '/home';
                            }
                            else{
                                alert('There is somthing wrong with your data!');
                                window.location.href = '/clean';
                            }
                        }
                    });
                }
                else if ($('input[name=select]:checked').val() == 'Geographical') {
                    console.log("成功进入geo");
                    $.ajax({
                        url: '/index/geography/',
                        type: 'POST',
                        data: {json_data: JSON.stringify(res)},
                        dataType: 'html',
                        success: function (res) {
                            if (res == "true") {
                                alert('success !');
                                window.location.href = '/geo/';
                            } else {
                                alert('Unknown error.');
                            }
                        },
                        error: function () {
                            alert('Internet error');
                        }
                    });
                }
                if (res[res.length - 1] == "") {
                    res.pop();
                }
                callback && callback(res);
            }
        });
    };
    fReader.onerror = function (evt) {
        alert("The file has changed,please select again.(Firefox)");
    };

    function checkEncoding(base64Str) {
        var str = atob(base64Str.split(";base64,")[1]);
        var encoding = jschardet.detect(str);
        encoding = encoding.encoding;
        console.log(encoding);
        if (encoding == "windows-1252") {
            encoding = "ANSI";
        }
        return encoding;
    }
};

function read_file() {
    $("input[name=csvfile]").csv2arr(function (res) {
    });
}