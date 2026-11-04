pub(super) fn elapsed_timestamp_ticks(begin: u64, end: u64, bits: u32) -> Option<u64> {
    if bits == 0 || bits > 64 {
        return None;
    }
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let elapsed = end.wrapping_sub(begin) & mask;
    (elapsed > 0).then_some(elapsed)
}

#[cfg(test)]
mod timestamp_tests {
    use super::elapsed_timestamp_ticks;

    #[test]
    fn elapsed_timestamp_respects_valid_bits_and_wrap() {
        assert_eq!(elapsed_timestamp_ticks(10, 17, 64), Some(7));
        assert_eq!(elapsed_timestamp_ticks(250, 5, 8), Some(11));
        assert_eq!(elapsed_timestamp_ticks(u64::MAX - 2, 2, 64), Some(5));
        assert_eq!(
            elapsed_timestamp_ticks(0xff00 | 250, 0xee00 | 5, 8),
            Some(11)
        );
        assert_eq!(elapsed_timestamp_ticks(17, 17, 64), None);
        assert_eq!(elapsed_timestamp_ticks(10, 17, 0), None);
        assert_eq!(elapsed_timestamp_ticks(10, 17, 65), None);
    }
}
