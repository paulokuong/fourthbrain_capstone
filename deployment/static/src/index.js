/*
Main page
=========
*/
'use strict';

import _ from 'lodash';
import axios from 'axios';
import dateformat from 'dateformat';
import 'bootstrap';
import 'bootstrap/js/dist/tooltip';
import 'selectize';
import randomColor from 'randomcolor';
import 'datatables.net';
import 'datatables.net-dt';
import gradient from 'gradient-color';
import 'bootstrap4-toggle';
import 'jquery-ui-bundle';
import Plotly from 'plotly.js-dist';
import Slider from 'bootstrap-slider';

/*
feature 3   0.067646  totalViewProducts
feature 8   0.062860  totalAddToCartQty
feature 4   0.055818    totalAddToCarts
feature 15  0.034540   productPriceMean
feature 9   0.033196          hourOfDay
feature 0   0.016931     uniqueSearches
feature 1   0.014983      totalSearches
feature 12  0.013691       has_campaign
feature 14  0.012890            browser
*/


class Index {
    constructor(env) {
      this.bars = [
        "total_view_products",
        "total_add_to_cart_qty",
        "total_add_to_carts",
        "product_price_mean",
        "hour_of_day",
        "unique_searches",
        "total_searches"
        // "has_campaign"
      ];
      this.last_40_event = '';
    }
    hookSlidingBars() {
      this.bars.forEach(function (item, index) {
        console.log(item, index);
        var mySlider = new Slider("#"+item, {
          formatter: function(value) {
            return 'Current value: ' + value;
          }
        });
      });
    }

    user_conversion_predict(){
      var query_string_arr = [];
      this.bars.forEach(function (item, index) {
        query_string_arr.push(item + '=' + $("#"+item).val());
      });
      var url = '/user_conversion_predict?' + query_string_arr.join('&');
      axios.get(url).then(function(res) {
        console.log(res);
        $("#convert").html(res.data["convert"]);
        $("#nonconvert").html(res.data["nonconvert"]);
      });
    }

    seq_user_conversion_predict(){
      var seq = $("#last_40_event").val();
      seq = seq.split("").reverse().join("");
      var url = '/user_conversion_lstm_predict?seq=' + seq;
      axios.get(url).then(function(res) {
        console.log(res);
        $("#seq_convert").html(res.data["predictions"][0]);
        $("#seq_nonconvert").html(res.data["predictions"][1]);
      });
    }

    set_var(){
      var padded = this.last_40_event.padStart(40, "0");
      var last_40 = padded.substring(padded.length - 40)
      this.last_40_event = last_40;
      console.log(last_40);
      $("#last_40_event").val(last_40);
      $("#seq_convert").html('predicting...');
      $("#seq_nonconvert").html('predicting...');
      this.seq_user_conversion_predict();
    }

    hook_seq_buttons(){
      var this_class = this;
      $("#view_product_button").click(function(){
        this_class.last_40_event += '1';
        this_class.set_var();
      });
      $("#add_to_cart_button").click(function(){
        this_class.last_40_event += '2';
        this_class.set_var();
      });
      $("#search_button").click(function(){
        this_class.last_40_event += '3';
        this_class.set_var();
      });
      $("#clear_button").click(function(){
        this_class.last_40_event = '';
        $("#last_40_event").val("");
        $("#seq_convert").textContent='';
        $("#seq_nonconvert").textContent='';
      });
    }
}

$(document).ready(function(e) {
    var i = new Index();
    i.hookSlidingBars();
    $("#predict_button").click(function(){
      i.user_conversion_predict();
    });
    i.user_conversion_predict();
    i.hook_seq_buttons();
});
