$(document).ready(function() {

    let lastcnt = 0;
    let cnt;
    chkNewScan();
  
    function chkNewScan() {
        countTodayScan();
        setTimeout(chkNewScan, 1000);
    }
  
    function countTodayScan() {
        $.ajax({
            url: '/countTodayScan',
            type: 'GET',
            dataType: 'json',
            success: function(data) {
                cnt = data.rowcount;
                if (cnt > lastcnt) {
                    reloadTable();
                }
  
                lastcnt = cnt;
            },
            error: function(result){
                console.log('no result!')
            }
        })
    }
  
    function reloadTable() {
        $.ajax({
            url: '/loadData',
            type: 'GET',
            dataType: 'json',
            success: function(response){
                var tr = $("#scandata");
                tr.empty();
                
                $.each(response, function(index, item) {
                    if (item.length > 0) {
                        for (let i = 0; i < item.length; i++) {
                            tr.append('<tr>'+
                                            '<td>'+item[i][1]+'</td>'+
                                            '<td>'+item[i][2]+'</td>'+
                                            '<td>'+item[i][3]+'</td>'+
                                            '<td>'+item[i][4]+'</td>'+
                                       '</tr>');
                        }
                    }
                });
            },
            error: function(result){
                console.log('data tidak tampil!')
            }
        });
    }
});