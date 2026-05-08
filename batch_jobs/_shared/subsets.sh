get_subsets() {
  local target="$1"
  local dataset="$2"
  local split_type="$3"

  case "${target}:${dataset}:${split_type}" in
    abs:deep4chem:random)
      echo "N00050_s42 N00100_s42 N00250_s42 N01000_s42 N04000_s42 N11817_s42"
      ;;
    abs:deep4chem:scaffold)
      echo "N00050_s42 N00100_s42 N00250_s42 N01000_s42 N04000_s42 N11816_s42"
      ;;
    em:deep4chem:random|em:deep4chem:scaffold)
      echo "N00050_s42 N00100_s42 N00250_s42 N01000_s42 N04000_s42 N11502_s42"
      ;;
    abs:chemfluor:random|abs:chemfluor:scaffold)
      echo "N00050_s42 N00100_s42 N00250_s42 N01000_s42 N03072_s42"
      ;;
    em:chemfluor:scaffold)
      echo "N00050_s42 N00100_s42 N00250_s42 N01000_s42 N03162_s42"
      ;;
    abs:dsscdb:random|abs:dsscdb:scaffold)
      echo "N00050_s42 N00100_s42 N00250_s42 N01000_s42 N01376_s42"
      ;;
    em:dsscdb:random|em:dsscdb:scaffold)
      echo "N00050_s42 N00100_s42 N00250_s42 N00552_s42"
      ;;
    abs:jeffries:random|abs:jeffries:scaffold)
      echo "N00020_s42 N00034_s42"
      ;;
    em:jeffries:random|em:jeffries:scaffold)
      echo "N00020_s42 N00034_s42"
      ;;
    em:jeffries:holdout)
      echo "N00038_s42"
      ;;
    *)
      echo "[ERROR] No subset list defined for ${target}:${dataset}:${split_type}" >&2
      return 1
      ;;
  esac
}
