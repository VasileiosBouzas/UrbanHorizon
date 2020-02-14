<?php 
$requests  = $_POST['kvcArray']['0'];

//echo sizeof($requests);
$requestsPolyR  = $_POST['kvcArray']['1'];
$requestsPolyA  = $_POST['kvcArray']['2'];
$user=$requests['0']['userID'];
$filename='filelog/'.strval($user).'.json';
// echo($filename);
$q['coordinates'] = $requests;
$q['polygonRemove']= $requestsPolyR;
$q['polygonAdd']= $requestsPolyA;
$fp = fopen($filename, 'w');
fwrite($fp, json_encode($q));
fclose($fp);
$result = shell_exec('C:/Python27/python D:/xampp/htdocs/SynthesisProject/SVFscript.py '.$filename);
echo $result;
?>
