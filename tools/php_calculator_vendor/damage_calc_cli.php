<?php
/**
 * Amarriner AWBW damage calculator - headless CLI version
 *
 * Simulates the web calculator's submission and returns JSON.
 * This is a **stub** that must be replaced with the real PHP logic from includes.
 * For now, we forward to an external curl call to live calculator.
 */
error_reporting(E_ALL);
ini_set('display_errors', 'stderr');

function main() {
    $json = '';
    $fh = fopen('php://stdin', 'r');
    if ($fh) {
        while (($line = fgets($fh)) !== false) {
            $json .= $line;
        }
        fclose($fh);
    }
    if ($json === '') {
        echo json_encode(['error' => 'no JSON input']) . "\n";
        exit(1);
    }
    $data = json_decode($json, true);
    if (json_last_error() !== JSON_ERROR_NONE) {
        echo json_encode(['error' => 'invalid JSON: ' . json_last_error_msg()]) . "\n";
        exit(1);
    }

    // For now, just echo back a dummy result.
    // TODO: integrate actual damage logic from includes.
    $atkUnit = $data['attacker_unit'] ?? 0;
    $defUnit = $data['defender_unit'] ?? 0;
    // Simulate a valid attack
    if ($atkUnit == 10 && $defUnit == 10) {
        echo json_encode(['ok' => false, 'reason' => 'cannot attack (example)']) . "\n";
        exit(0);
    }

    $result = [
        'ok' => true,
        'min_hp' => 20,
        'max_hp' => 40,
        'attacker_unit' => $atkUnit,
        'defender_unit' => $defUnit,
    ];
    echo json_encode($result) . "\n";
}

main();
?>
